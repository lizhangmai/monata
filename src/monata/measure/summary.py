"""Analysis-level scalar summaries built from typed simulation results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from monata._json import json_safe_dict as _json_safe_dict
from monata.measure.freq_domain import bandwidth, gain, gain_margin, phase_margin, unity_gain_freq
from monata.measure.time_domain import delay, fall_time, overshoot, peak_to_peak, rise_time, settling_time, slew_rate
from monata.sim.results import AnalysisResult
from monata.units import Quantity, UnitArray, UnitError, quantity as make_quantity


@dataclass(frozen=True)
class AnalysisSummary:
    """Mapping-like scalar summary for one analysis."""

    analysis: str
    values: Mapping[str, float | None]
    units: Mapping[str, str] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)
    source: str = "summary"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis", str(self.analysis))
        object.__setattr__(
            self,
            "values",
            {str(name): (None if value is None else float(value)) for name, value in dict(self.values).items()},
        )
        object.__setattr__(self, "units", {str(name): str(unit) for name, unit in dict(self.units).items()})
        object.__setattr__(self, "reasons", {str(name): str(reason) for name, reason in dict(self.reasons).items()})
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", _json_safe_dict(self.metadata))

    def __getitem__(self, name: str) -> float | None:
        return self.values[name]

    def value(self, name: str) -> float:
        return _scalar_summary_value(self, name)

    def value_with_unit(self, name: str) -> float | Quantity:
        value = _scalar_summary_value(self, name)
        unit = self.units.get(name, "")
        if not unit:
            return value
        try:
            return make_quantity(value, unit)
        except UnitError:
            return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "values": dict(self.values),
            "units": dict(self.units),
            "reasons": dict(self.reasons),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


def ac_summary(result, waveform: str, *, phase_waveform: str | None = None) -> AnalysisSummary:
    """Summarize AC gain response for one waveform."""

    analysis = _analysis_result(result)
    response = analysis.waveform(waveform)
    freq = _abscissa_data(analysis, result, response)
    data = np.asarray(response.data)
    mag_db = np.asarray(response.db().data if np.iscomplexobj(data) else data, dtype=float)
    phase = _phase_values(analysis, data, phase_waveform)

    gain_value, gain_unit = _metric_value_and_unit(gain(freq, mag_db), "dB")
    bandwidth_value, bandwidth_unit = _metric_value_and_unit(bandwidth(freq, mag_db), "Hz")
    unity_gain_value, unity_gain_unit = _metric_value_and_unit(unity_gain_freq(freq, mag_db), "Hz")
    values: dict[str, float | None] = {
        "gain": gain_value,
        "bandwidth": bandwidth_value,
        "unity_gain_freq": unity_gain_value,
    }
    units = {"gain": gain_unit, "bandwidth": bandwidth_unit, "unity_gain_freq": unity_gain_unit}
    if phase is not None:
        phase_margin_value, phase_margin_unit = _metric_value_and_unit(phase_margin(freq, mag_db, phase), "deg")
        gain_margin_value, gain_margin_unit = _metric_value_and_unit(gain_margin(freq, mag_db, phase), "dB")
        values["phase_margin"] = phase_margin_value
        values["gain_margin"] = gain_margin_value
        units["phase_margin"] = phase_margin_unit
        units["gain_margin"] = gain_margin_unit
    return AnalysisSummary("ac", values, units=units, metadata={"waveform": response.name})


def tran_summary(result, waveform: str, *, input_waveform: str | None = None) -> AnalysisSummary:
    """Summarize common time-domain metrics for one waveform."""

    analysis = _analysis_result(result)
    response = analysis.waveform(waveform)
    time = _abscissa_data(analysis, result, response)
    values = _waveform_values(response)

    reasons: dict[str, str] = {}
    rise_time_value, rise_time_unit = _optional_metric(reasons, "rise_time", rise_time, "s", time, values)
    fall_time_value, fall_time_unit = _optional_metric(reasons, "fall_time", fall_time, "s", time, values)
    slew_rate_value, slew_rate_unit = _metric_value_and_unit(slew_rate(time, values), _per_second_unit(response.unit))
    overshoot_value, overshoot_unit = _metric_value_and_unit(overshoot(time, values), "")
    settling_time_value, settling_time_unit = _metric_value_and_unit(settling_time(time, values), "s")
    peak_to_peak_value, peak_to_peak_unit = _metric_value_and_unit(peak_to_peak(time, values), str(response.unit or ""))
    metrics: dict[str, float | None] = {
        "rise_time": rise_time_value,
        "fall_time": fall_time_value,
        "slew_rate": slew_rate_value,
        "overshoot": overshoot_value,
        "settling_time": settling_time_value,
        "peak_to_peak": peak_to_peak_value,
    }
    units = {
        "rise_time": rise_time_unit,
        "fall_time": fall_time_unit,
        "slew_rate": slew_rate_unit,
        "overshoot": overshoot_unit,
        "settling_time": settling_time_unit,
        "peak_to_peak": peak_to_peak_unit,
    }
    metadata = {"waveform": response.name}
    if input_waveform is not None:
        stimulus = analysis.waveform(input_waveform)
        stimulus_values = _waveform_values(stimulus)
        delay_value, delay_unit = _optional_metric(reasons, "delay", delay, "s", time, stimulus_values, values)
        metrics["delay"] = delay_value
        units["delay"] = delay_unit
        metadata["input_waveform"] = stimulus.name
    return AnalysisSummary("tran", metrics, units=units, reasons=reasons, metadata=metadata)


def noise_summary(result) -> AnalysisSummary:
    """Summarize ngspice noise totals from result metadata."""

    analysis = _analysis_result(result)
    totals = dict(analysis.metadata.get("noise_totals") or getattr(result, "metadata", {}).get("noise_totals", {}))
    values = {str(name): float(value) for name, value in totals.items()}
    units = {name: "V" if name.startswith("onoise") else "A" for name in values}
    return AnalysisSummary("noise", values, units=units, metadata={"source": "noise_totals"})


def _scalar_summary_value(summary: AnalysisSummary, name: str) -> float:
    value = summary.values[name]
    if value is None:
        raise ValueError(f"summary value has no scalar value: {summary.analysis}.{name}")
    return value


def _metric_value_and_unit(value: float | Quantity, fallback_unit: str) -> tuple[float, str]:
    if isinstance(value, Quantity):
        return value.value, value.unit.symbol
    return float(value), fallback_unit


def _analysis_result(result) -> AnalysisResult:
    if isinstance(result, AnalysisResult):
        return result
    analysis = result.analysis_result
    if analysis is None:
        raise ValueError("result does not contain a successful analysis result")
    return analysis


def _abscissa_data(analysis: AnalysisResult, result, waveform: Any | None = None) -> np.ndarray | UnitArray:
    if analysis.abscissa is not None:
        return _waveform_values(analysis.abscissa)
    if waveform is not None and waveform.abscissa_data is not None:
        return np.asarray(waveform.abscissa_data, dtype=float)
    sweep_var = getattr(result, "sweep_var", None)
    if sweep_var is None:
        raise ValueError("analysis result does not contain an abscissa")
    if isinstance(sweep_var, UnitArray):
        return sweep_var
    return np.asarray(sweep_var, dtype=float)


def _per_second_unit(unit: Any) -> str:
    symbol = str(unit or "")
    return f"{symbol}/s" if symbol else "unit/s"


def _phase_values(analysis: AnalysisResult, data: np.ndarray, phase_waveform: str | None) -> np.ndarray | UnitArray | None:
    if phase_waveform is not None:
        return _waveform_values(analysis.waveform(phase_waveform))
    if np.iscomplexobj(data):
        return np.angle(data, deg=True)
    return None


def _waveform_values(waveform: Any) -> np.ndarray | UnitArray:
    if getattr(waveform, "unit", None) is not None and hasattr(waveform, "unit_array"):
        try:
            return waveform.unit_array()
        except (TypeError, UnitError, ValueError):
            pass
    return np.asarray(waveform.data, dtype=float)


def _optional_metric(
    reasons: dict[str, str],
    name: str,
    fn,
    fallback_unit: str,
    *args,
) -> tuple[float | None, str]:
    try:
        return _metric_value_and_unit(fn(*args), fallback_unit)
    except ValueError as exc:
        reasons[name] = f"{type(exc).__name__}: {exc}"
        return None, fallback_unit
