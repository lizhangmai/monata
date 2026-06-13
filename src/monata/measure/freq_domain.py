"""Frequency-domain measurement primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from monata.units import Quantity, UnitArray, UnitError, unit


@dataclass(frozen=True)
class BodeTrace:
    """Plotting-neutral Bode magnitude and phase arrays."""

    frequency: np.ndarray
    gain_db: np.ndarray
    phase: np.ndarray
    phase_unit: Literal["deg", "rad"] = "deg"

    def to_arrays(
        self,
        *,
        copy: bool = False,
        phase_unit: Literal["deg", "rad"] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return column-oriented arrays for tabular export or custom plotting."""

        target_phase_unit = self.phase_unit if phase_unit is None else _valid_phase_unit(phase_unit)
        phase_column = "phase_deg" if target_phase_unit == "deg" else "phase_rad"
        phase = _phase_values(self.phase, source_unit=self.phase_unit, target_unit=target_phase_unit)
        return {
            "frequency": _array_column(self.frequency, copy=copy),
            "gain_db": _array_column(self.gain_db, copy=copy),
            phase_column: _array_column(phase, copy=copy),
        }


def bode_trace(
    frequency: Any,
    response: Any,
    *,
    phase_unit: Literal["deg", "rad"] = "deg",
    unwrap_phase: bool = False,
) -> BodeTrace:
    """Convert a complex frequency response into Bode gain and phase arrays."""

    frequency_array = _frequency_array(frequency)
    response_array = np.asarray(response)
    if frequency_array.ndim != 1 or response_array.ndim != 1:
        raise ValueError("Bode trace requires one-dimensional frequency and response arrays")
    if frequency_array.shape != response_array.shape:
        raise ValueError("Bode frequency and response arrays must have the same shape")
    if frequency_array.size == 0:
        raise ValueError("Bode trace requires at least one point")
    if np.any(frequency_array <= 0):
        raise ValueError("Bode frequency values must be positive")
    if frequency_array.size > 1 and np.any(np.diff(frequency_array) <= 0):
        raise ValueError("Bode frequency values must be strictly increasing")
    if phase_unit not in {"deg", "rad"}:
        raise ValueError("phase_unit must be 'deg' or 'rad'")

    with np.errstate(divide="ignore"):
        gain_db = 20 * np.log10(np.abs(response_array))
    phase = np.angle(response_array)
    if unwrap_phase:
        phase = np.unwrap(phase)
    if phase_unit == "deg":
        phase = np.degrees(phase)
    return BodeTrace(frequency_array, gain_db, phase, phase_unit)


def gain(freq: Any, mag_db: Any) -> float:
    _freq, mag = _frequency_gain_arrays(freq, mag_db)
    return float(mag[0])


def bandwidth(freq: Any, mag_db: Any, gain_drop: float = -3.0) -> float | Quantity:
    freq_array, mag = _frequency_gain_arrays(freq, mag_db)
    value = _bandwidth_value(freq_array, mag, gain_drop=gain_drop)
    return _frequency_result(freq, value)


def gain_bandwidth_product(freq: Any, mag_db: Any) -> float | Quantity:
    freq_array, mag = _frequency_gain_arrays(freq, mag_db)
    dc_gain_linear = 10 ** (mag[0] / 20)
    bw = _bandwidth_value(freq_array, mag, gain_drop=-3.0)
    return _frequency_result(freq, float(dc_gain_linear * bw))


def phase_margin(freq: Any, mag_db: Any, phase_deg: Any) -> float:
    freq_array, mag = _frequency_gain_arrays(freq, mag_db)
    phase = _matching_phase_array(freq_array, phase_deg)
    ugf = _interp_freq_at_gain(freq_array, mag, 0.0)
    phase_at_ugf = float(np.interp(np.log10(ugf), np.log10(freq_array), phase))
    return 180.0 + phase_at_ugf


def gain_margin(freq: Any, mag_db: Any, phase_deg: Any) -> float:
    freq_array, mag = _frequency_gain_arrays(freq, mag_db)
    phase = _matching_phase_array(freq_array, phase_deg)
    for i in range(len(phase) - 1):
        if phase[i] > -180.0 >= phase[i + 1]:
            frac = (-180.0 - phase[i]) / (phase[i + 1] - phase[i])
            f_180 = freq_array[i] * (freq_array[i + 1] / freq_array[i]) ** frac
            gain_at_f180 = float(np.interp(np.log10(f_180), np.log10(freq_array), mag))
            return -gain_at_f180
    return float("inf")


def unity_gain_freq(freq: Any, mag_db: Any) -> float | Quantity:
    freq_array, mag = _frequency_gain_arrays(freq, mag_db)
    return _frequency_result(freq, _interp_freq_at_gain(freq_array, mag, 0.0))


def rejection_at_freq(freq: Any, mag_db: Any, f_target: Any) -> float:
    freq_array, mag = _frequency_gain_arrays(freq, mag_db)
    target = _target_frequency_scalar(f_target)
    if target < freq_array[0] or target > freq_array[-1]:
        raise ValueError("target frequency must be within sampled frequency range")
    return float(np.interp(np.log10(target), np.log10(freq_array), mag))


def _interp_freq_at_gain(freq: np.ndarray, mag_db: np.ndarray, target_db: float) -> float:
    log_freq = np.log10(freq)
    for i in range(len(mag_db) - 1):
        lower = float(mag_db[i])
        upper = float(mag_db[i + 1])
        if lower == target_db:
            return float(freq[i])
        if upper == target_db:
            return float(freq[i + 1])
        if (lower - target_db) * (upper - target_db) < 0:
            frac = (target_db - lower) / (upper - lower)
            log_f = log_freq[i] + frac * (log_freq[i + 1] - log_freq[i])
            return 10 ** log_f
    raise ValueError(f"Gain level {target_db} dB not crossed in data")


def _bandwidth_value(freq: np.ndarray, mag_db: np.ndarray, *, gain_drop: float) -> float:
    dc_gain = mag_db[0]
    target = dc_gain + gain_drop
    return _interp_freq_at_gain(freq, mag_db, target)


def _frequency_result(frequency: Any, value: float) -> float | Quantity:
    if _frequency_unit_values(frequency) is None:
        return float(value)
    return Quantity(float(value), unit("Hz"))


def _array_column(values: np.ndarray, *, copy: bool) -> np.ndarray:
    if copy:
        return np.array(values, copy=True)
    return np.asarray(values)


def _phase_values(
    values: np.ndarray,
    *,
    source_unit: Literal["deg", "rad"],
    target_unit: Literal["deg", "rad"],
) -> np.ndarray:
    if target_unit == source_unit:
        return np.asarray(values)
    if target_unit == "deg":
        return np.degrees(values)
    return np.radians(values)


def _valid_phase_unit(phase_unit: Literal["deg", "rad"]) -> Literal["deg", "rad"]:
    if phase_unit not in {"deg", "rad"}:
        raise ValueError("phase_unit must be 'deg' or 'rad'")
    return phase_unit


def _frequency_gain_arrays(freq: Any, mag_db: Any) -> tuple[np.ndarray, np.ndarray]:
    freq_array = _frequency_array(freq)
    mag = np.asarray(mag_db, dtype=float)
    if freq_array.ndim != 1 or mag.ndim != 1:
        raise ValueError("frequency-domain measurements require one-dimensional arrays")
    if freq_array.shape != mag.shape:
        raise ValueError("frequency and magnitude arrays must have the same shape")
    if freq_array.size == 0:
        raise ValueError("frequency-domain measurements require at least one point")
    if not np.all(np.isfinite(freq_array)):
        raise ValueError("frequency values must be finite")
    if np.any(freq_array <= 0):
        raise ValueError("frequency values must be positive")
    if freq_array.size > 1 and np.any(np.diff(freq_array) <= 0):
        raise ValueError("frequency values must be strictly increasing")
    return freq_array, mag


def _matching_phase_array(freq: np.ndarray, phase_deg: Any) -> np.ndarray:
    phase = _phase_degree_array(phase_deg)
    if phase.ndim != 1 or phase.shape != freq.shape:
        raise ValueError("frequency and phase arrays must have the same shape")
    return phase


def _phase_degree_array(phase: Any) -> np.ndarray:
    try:
        if isinstance(phase, UnitArray):
            return np.asarray(phase.to("deg").values, dtype=float)
        if isinstance(phase, Quantity):
            return np.asarray(phase.to("deg").value, dtype=float)
    except UnitError as exc:
        raise ValueError("phase inputs must use angle-compatible units") from exc
    return np.asarray(phase, dtype=float)


def _frequency_array(frequency: Any) -> np.ndarray:
    unit_values = _frequency_unit_values(frequency)
    if unit_values is not None:
        return np.asarray(unit_values, dtype=float)
    return np.asarray(frequency, dtype=float)


def _frequency_scalar(frequency: Any) -> float:
    unit_values = _frequency_unit_values(frequency)
    if unit_values is not None:
        values = np.asarray(unit_values, dtype=float)
        if values.ndim == 0:
            return float(values)
        if values.shape == (1,):
            return float(values[0])
        raise ValueError("target frequency must be scalar")
    return float(frequency)


def _target_frequency_scalar(frequency: Any) -> float:
    target = _frequency_scalar(frequency)
    if not np.isfinite(target):
        raise ValueError("target frequency must be finite")
    if target <= 0:
        raise ValueError("target frequency must be positive")
    return target


def _frequency_unit_values(frequency: Any) -> Any | None:
    try:
        if isinstance(frequency, UnitArray):
            return frequency.to("Hz").values
        if isinstance(frequency, Quantity):
            return frequency.to("Hz").value
    except UnitError as exc:
        raise ValueError("frequency inputs must use frequency-compatible units") from exc
    return None
