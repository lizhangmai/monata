"""monata.measure — waveform measurement primitives."""

from typing import TYPE_CHECKING

from monata.measure.calculus import (
    exact_finite_difference_coefficients,
    finite_difference_coefficients,
    finite_difference_derivative,
    finite_difference_stencil,
    simple_derivative,
)
from monata.measure.time_domain import (
    rise_time, fall_time, delay, slew_rate, overshoot,
    settling_time, cross, period, duty_cycle, peak_to_peak,
)
from monata.measure.freq_domain import (
    BodeTrace, bode_trace, gain, bandwidth, gain_bandwidth_product, phase_margin,
    gain_margin, unity_gain_freq, rejection_at_freq,
)
from monata.measure.statistics import histogram, sigma_yield, worst_case, sensitivity
from monata.measure.result import MeasureNotFoundError, MeasureResult, MeasureSet
from monata.measure.spec import Spec, SpecResult, SpecTable

if TYPE_CHECKING:
    from monata.measure.summary import AnalysisSummary, ac_summary, noise_summary, tran_summary

_SUMMARY_EXPORTS = {"AnalysisSummary", "ac_summary", "noise_summary", "tran_summary"}

__all__ = [
    "exact_finite_difference_coefficients",
    "finite_difference_coefficients",
    "finite_difference_derivative",
    "finite_difference_stencil",
    "simple_derivative",
    "rise_time", "fall_time", "delay", "slew_rate", "overshoot",
    "settling_time", "cross", "period", "duty_cycle", "peak_to_peak",
    "BodeTrace", "bode_trace",
    "gain", "bandwidth", "gain_bandwidth_product", "phase_margin",
    "gain_margin", "unity_gain_freq", "rejection_at_freq",
    "histogram", "sigma_yield", "worst_case", "sensitivity",
    "MeasureNotFoundError", "MeasureResult", "MeasureSet",
    "AnalysisSummary", "ac_summary", "noise_summary", "tran_summary",
    "Spec", "SpecResult", "SpecTable",
]


def __getattr__(name: str) -> object:
    if name in _SUMMARY_EXPORTS:
        from importlib import import_module

        summary = import_module("monata.measure.summary")
        value = getattr(summary, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'monata.measure' has no attribute {name!r}")
