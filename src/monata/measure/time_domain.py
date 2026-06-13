"""Time-domain measurement primitives. Pure functions: (t, v) -> scalar."""

from __future__ import annotations

from typing import Any

import numpy as np

from monata.units import Quantity, Unit, UnitArray, UnitError, unit, unit_value


def rise_time(t: Any, v: Any, low: float = 0.1, high: float = 0.9) -> float | Quantity:
    low_value, high_value = _transition_fraction_bounds(low, high, "rise_time")
    t_values, v_values = _validated_time_series(t, v)
    v_min, v_max = v_values.min(), v_values.max()
    v_low = v_min + low_value * (v_max - v_min)
    v_high = v_min + high_value * (v_max - v_min)
    t_low = _interp_crossing(t_values, v_values, v_low, edge="rising")
    t_high = _interp_crossing(t_values, v_values, v_high, edge="rising")
    return _time_result(t, t_high - t_low)


def fall_time(t: Any, v: Any, low: float = 0.1, high: float = 0.9) -> float | Quantity:
    low_value, high_value = _transition_fraction_bounds(low, high, "fall_time")
    t_values, v_values = _validated_time_series(t, v)
    v_min, v_max = v_values.min(), v_values.max()
    v_high = v_min + high_value * (v_max - v_min)
    v_low = v_min + low_value * (v_max - v_min)
    t_high = _interp_crossing(t_values, v_values, v_high, edge="falling")
    t_low = _interp_crossing(t_values, v_values, v_low, edge="falling")
    return _time_result(t, t_low - t_high)


def delay(
    t: Any,
    v_in: Any,
    v_out: Any,
    threshold: float = 0.5,
    *,
    input_edge: str = "rising",
    output_edge: str = "rising",
    input_n: int = 1,
    output_n: int = 1,
) -> float | Quantity:
    threshold_value = _fraction(threshold, "delay threshold")
    t_in, v_in_values = _validated_time_series(t, v_in)
    t_out, v_out_values = _validated_time_series(t, v_out)
    v_in_min, v_in_max = v_in_values.min(), v_in_values.max()
    v_out_min, v_out_max = v_out_values.min(), v_out_values.max()
    th_in = v_in_min + threshold_value * (v_in_max - v_in_min)
    th_out = v_out_min + threshold_value * (v_out_max - v_out_min)
    in_crossing = _nth_crossing(t_in, v_in_values, th_in, edge=input_edge, n=input_n)
    out_crossing = _nth_crossing(t_out, v_out_values, th_out, edge=output_edge, n=output_n)
    return _time_result(t, out_crossing - in_crossing)


def slew_rate(t: Any, v: Any) -> float | Quantity:
    t_values, v_values = _validated_time_series(t, v)
    dv_dt = np.gradient(v_values, t_values)
    value = float(np.max(np.abs(dv_dt)))
    output_unit = _slew_rate_unit(t, v)
    if output_unit is None:
        return value
    return Quantity(value, output_unit)


def overshoot(t: Any, v: Any, final_value: Any | None = None) -> float:
    _t_values, v_values = _validated_time_series(t, v)
    final = float(v_values[-1]) if final_value is None else _value_scalar(final_value, v, "final_value")
    if final == 0:
        return 0.0
    peak = v_values.max()
    return float((peak - final) / abs(final))


def settling_time(t: Any, v: Any, final_value: Any | None = None, tolerance: float = 0.01) -> float | Quantity:
    tolerance_value = _nonnegative_finite(tolerance, "settling tolerance")
    t_values, v_values = _validated_time_series(t, v)
    final = float(v_values[-1]) if final_value is None else _value_scalar(final_value, v, "final_value")
    band = abs(final) * tolerance_value
    outside = np.where(np.abs(v_values - final) > band)[0]
    if len(outside) == 0:
        return _time_result(t, 0.0)
    return _time_result(t, _settling_boundary_time(t_values, v_values, final, band, int(outside[-1])))


def cross(
    t: Any,
    v: Any,
    threshold: Any,
    edge: str = "rising",
    n: int = 1,
    *,
    start: Any | None = None,
    stop: Any | None = None,
) -> float | Quantity:
    t_values, v_values = _validated_time_series(t, v)
    level = _value_scalar(threshold, v, "threshold")
    start_value = _optional_time_scalar(start, "start")
    stop_value = _optional_time_scalar(stop, "stop")
    return _time_result(
        t,
        _nth_crossing(
            t_values,
            v_values,
            level,
            edge=edge,
            n=n,
            start=start_value,
            stop=stop_value,
        ),
    )


def period(t: Any, v: Any, threshold: Any | None = None) -> float | Quantity:
    _t_values, v_values = _validated_time_series(t, v)
    level = float((v_values.max() + v_values.min()) / 2) if threshold is None else _value_scalar(threshold, v, "threshold")
    t1 = cross(t, v_values, level, edge="rising", n=1)
    t2 = cross(t, v_values, level, edge="rising", n=2)
    if isinstance(t1, Quantity) and isinstance(t2, Quantity):
        return t2 - t1
    return float(t2) - float(t1)


def duty_cycle(t: Any, v: Any, threshold: Any | None = None) -> float:
    t_values, v_values = _validated_time_series(t, v)
    level = float((v_values.max() + v_values.min()) / 2) if threshold is None else _value_scalar(threshold, v, "threshold")
    high_time = 0.0
    for t0, t1, v0, v1 in zip(t_values[:-1], t_values[1:], v_values[:-1], v_values[1:], strict=True):
        high_time += _interval_time_above(float(t0), float(t1), float(v0), float(v1), level)
    return high_time / float(t_values[-1] - t_values[0])


def peak_to_peak(t: Any, v: Any) -> float | Quantity:
    _t_values, v_values = _validated_time_series(t, v)
    value = float(v_values.max() - v_values.min())
    value_unit = _value_unit(v)
    if value_unit is None:
        return value
    return Quantity(value, value_unit)


def _validated_time_series(t: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
    t_values = _time_array(t)
    v_values = np.asarray(v, dtype=float)
    if t_values.ndim != 1 or v_values.ndim != 1:
        raise ValueError("time-domain measurements require one-dimensional arrays")
    if t_values.shape[0] != v_values.shape[0]:
        raise ValueError("time-domain measurements require matching time and value lengths")
    if t_values.shape[0] < 2:
        raise ValueError("time-domain measurements require at least two samples")
    if np.any(np.diff(t_values) <= 0):
        raise ValueError("time-domain measurements require strictly increasing time values")
    return t_values, v_values


def _transition_fraction_bounds(low: float, high: float, label: str) -> tuple[float, float]:
    low_value = _fraction(low, f"{label} low")
    high_value = _fraction(high, f"{label} high")
    if low_value >= high_value:
        raise ValueError(f"{label} low must be less than high")
    return low_value, high_value


def _fraction(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return number


def _nonnegative_finite(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _time_result(t: Any, value: float) -> float | Quantity:
    if _time_unit(t) is None:
        return float(value)
    return Quantity(float(value), unit("s"))


def _slew_rate_unit(t: Any, v: Any) -> Unit | None:
    value_unit = _value_unit(v)
    time_has_unit = _time_unit(t) is not None
    if value_unit is None and not time_has_unit:
        return None
    if value_unit is None:
        value_unit = unit_value(1).unit
    return value_unit / unit("s")


def _time_unit(t: Any) -> Unit | None:
    if isinstance(t, (Quantity, UnitArray)):
        return t.unit
    return None


def _value_unit(v: Any) -> Unit | None:
    if isinstance(v, UnitArray):
        return v.unit
    return None


def _value_scalar(value: Any, v: Any, label: str) -> float:
    value_unit = _value_unit(v)
    if isinstance(value, Quantity) and value_unit is not None:
        try:
            return value.to(value_unit).value
        except UnitError as exc:
            raise ValueError(f"{label} must use value-compatible units") from exc
    return float(value)


def _time_array(t: Any) -> np.ndarray:
    unit_values = _time_unit_values(t)
    if unit_values is not None:
        return np.asarray(unit_values, dtype=float)
    return np.asarray(t, dtype=float)


def _time_unit_values(t: Any) -> Any | None:
    from monata.units import Quantity, UnitArray, UnitError

    try:
        if isinstance(t, UnitArray):
            return t.to("s").values
        if isinstance(t, Quantity):
            return t.to("s").value
    except UnitError as exc:
        raise ValueError("time inputs must use time-compatible units") from exc
    return None


def _optional_time_scalar(value: Any | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, Quantity):
        try:
            return float(value.to("s").value)
        except UnitError as exc:
            raise ValueError(f"{label} must use time-compatible units") from exc
    return _finite_float(value, label)


def _settling_boundary_time(
    t_values: np.ndarray,
    v_values: np.ndarray,
    final: float,
    band: float,
    last_outside_index: int,
) -> float:
    if last_outside_index >= len(t_values) - 1:
        return float(t_values[last_outside_index])
    lower = final - band
    upper = final + band
    v0 = float(v_values[last_outside_index])
    v1 = float(v_values[last_outside_index + 1])
    if v0 < lower:
        target = lower
    elif v0 > upper:
        target = upper
    else:
        return float(t_values[last_outside_index])
    if v1 == v0:
        return float(t_values[last_outside_index + 1])
    fraction = (target - v0) / (v1 - v0)
    if not 0.0 <= fraction <= 1.0:
        return float(t_values[last_outside_index + 1])
    return float(t_values[last_outside_index] + fraction * (t_values[last_outside_index + 1] - t_values[last_outside_index]))


def _interval_time_above(t0: float, t1: float, v0: float, v1: float, threshold: float) -> float:
    start_above = v0 > threshold
    end_above = v1 > threshold
    if start_above and end_above:
        return t1 - t0
    if start_above == end_above:
        return 0.0
    crossing = t0 + (threshold - v0) / (v1 - v0) * (t1 - t0)
    if start_above:
        return crossing - t0
    return t1 - crossing


def _validated_edge(edge: str) -> str:
    edge_name = str(edge).lower()
    if edge_name not in {"rising", "falling", "either"}:
        raise ValueError("edge must be 'rising', 'falling', or 'either'")
    return edge_name


def _crossing_time(
    t0: float,
    t1: float,
    v0: float,
    v1: float,
    threshold: float,
    edge: str,
) -> float | None:
    if v1 == v0:
        return None
    rising = v0 < threshold <= v1
    falling = v0 > threshold >= v1
    if edge == "rising" and not rising:
        return None
    if edge == "falling" and not falling:
        return None
    if edge == "either" and not (rising or falling):
        return None
    frac = (threshold - v0) / (v1 - v0)
    return float(t0 + frac * (t1 - t0))


def _interp_crossing(t: np.ndarray, v: np.ndarray, threshold: float, edge: str = "rising") -> float:
    t_values, v_values = _validated_time_series(t, v)
    return _nth_crossing(t_values, v_values, threshold, edge=edge, n=1)


def _nth_crossing(
    t_values: np.ndarray,
    v_values: np.ndarray,
    threshold: float,
    *,
    edge: str,
    n: int,
    start: float | None = None,
    stop: float | None = None,
) -> float:
    edge_name = _validated_edge(edge)
    if n < 1:
        raise ValueError("crossing index n must be positive")
    start_value = float(t_values[0]) if start is None else start
    stop_value = float(t_values[-1]) if stop is None else stop
    if start_value > stop_value:
        raise ValueError("crossing start must be <= stop")
    count = 0
    for t0, t1, v0, v1 in zip(t_values[:-1], t_values[1:], v_values[:-1], v_values[1:], strict=True):
        if float(t1) < start_value or float(t0) > stop_value:
            continue
        crossing = _crossing_time(float(t0), float(t1), float(v0), float(v1), threshold, edge_name)
        if crossing is not None and not start_value <= crossing <= stop_value:
            continue
        if crossing is not None:
            count += 1
            if count == n:
                return crossing
    raise ValueError(f"Crossing #{n} ({edge_name}) at threshold={threshold} not found")
