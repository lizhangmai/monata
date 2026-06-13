"""Plotting helpers for typed simulation results."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Literal, Mapping

import numpy as np

from monata.sim.results import AnalysisResult, Waveform

_BODE_PHASE_TICKS = {
    "deg": (
        np.array([-180.0, -90.0, 0.0, 90.0, 180.0]),
        ("-180", "-90", "0", "90", "180"),
    ),
    "rad": (
        np.array([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi]),
        (r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"),
    ),
}


def plot_waveform(
    waveform: Waveform,
    *plot_args: Any,
    ax: Any = None,
    label: str | None = None,
    **plot_kwargs: Any,
) -> Any:
    """Plot one waveform with matplotlib."""

    axis = _pyplot().subplots()[1] if ax is None else ax
    x_values = _waveform_x_values(waveform)
    line_kwargs = dict(plot_kwargs)
    if label is not None:
        line_kwargs["label"] = label
    else:
        line_kwargs.setdefault("label", waveform.display_name or waveform.name)
    axis.plot(x_values, waveform.data, *plot_args, **line_kwargs)
    axis.set_ylabel(str(waveform.unit or waveform.quantity or waveform.name))
    axis.set_xlabel(waveform.abscissa_name or "index")
    return axis


def plot_bode(
    result: AnalysisResult,
    name: str,
    *,
    axes: tuple[Any, Any] | None = None,
    magnitude_kwargs: Mapping[str, Any] | None = None,
    phase_kwargs: Mapping[str, Any] | None = None,
    phase_unit: Literal["deg", "rad"] = "deg",
    **plot_kwargs: Any,
) -> tuple[Any, Any]:
    """Plot Bode magnitude in dB and phase in the requested unit for a complex waveform."""

    if axes is None:
        magnitude_axis, phase_axis = _pyplot().subplots(2, 1, sharex=True)[1]
    else:
        magnitude_axis, phase_axis = axes
    waveform = result.waveform(name)
    trace = result.bode_trace(name, phase_unit=phase_unit)
    magnitude_axis.semilogx(trace.frequency, trace.gain_db, **_merged_plot_kwargs(plot_kwargs, magnitude_kwargs))
    phase_axis.semilogx(trace.frequency, trace.phase, **_merged_plot_kwargs(plot_kwargs, phase_kwargs))
    magnitude_axis.set_ylabel("dB")
    phase_axis.set_ylabel(trace.phase_unit)
    phase_axis.set_xlabel(_waveform_x_label(waveform))
    _style_bode_axes(magnitude_axis, phase_axis, trace.phase_unit)
    return magnitude_axis, phase_axis


def plot_analysis(
    result: AnalysisResult,
    names: list[str] | tuple[str, ...] | None = None,
    *plot_args: Any,
    ax: Any = None,
    labels: Mapping[str, str] | None = None,
    **plot_kwargs: Any,
) -> Any:
    """Plot one or more waveforms from an AnalysisResult on a shared axis."""

    axis = _pyplot().subplots()[1] if ax is None else ax
    selected = tuple(names or result.waveforms.keys())
    waveforms = [result.waveform(name) for name in selected]
    for name in selected:
        waveform = result.waveform(name)
        x_data = _waveform_x_values(waveform)
        line_kwargs = dict(plot_kwargs)
        line_kwargs.setdefault("label", _plot_label(name, waveform, labels))
        axis.plot(x_data, waveform.data, *plot_args, **line_kwargs)
    axis.set_xlabel(_shared_xlabel(waveforms))
    y_label = _shared_ylabel(waveforms)
    if y_label:
        axis.set_ylabel(y_label)
    if len(selected) > 1:
        axis.legend()
    return axis


def _pyplot() -> Any:
    try:
        return import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise RuntimeError(
            'plotting requires matplotlib; install it with `python -m pip install "monata[plot]"`'
        ) from exc


def _shared_ylabel(waveforms: list[Waveform]) -> str:
    labels = {str(waveform.unit or waveform.quantity or "") for waveform in waveforms}
    labels.discard("")
    return labels.pop() if len(labels) == 1 else ""


def _shared_xlabel(waveforms: list[Waveform]) -> str:
    labels = {_waveform_x_label(waveform) for waveform in waveforms}
    return labels.pop() if len(labels) == 1 else "index"


def _waveform_x_values(waveform: Waveform, *, one_based: bool = False) -> np.ndarray:
    if waveform.abscissa_data is not None:
        return np.asarray(waveform.abscissa_data)
    start = 1 if one_based else 0
    stop = waveform.data.size + 1 if one_based else waveform.data.size
    return np.arange(start, stop)


def _waveform_x_label(waveform: Waveform) -> str:
    if waveform.abscissa_data is not None and waveform.abscissa_name is not None:
        return waveform.abscissa_name
    return "index"


def _merged_plot_kwargs(
    base: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {**dict(base), **dict(override or {})}


def _style_bode_axes(magnitude_axis: Any, phase_axis: Any, phase_unit: Literal["deg", "rad"]) -> None:
    for axis in (magnitude_axis, phase_axis):
        axis.grid(True)
        axis.grid(True, which="minor")
    phase_ticks, phase_tick_labels = _BODE_PHASE_TICKS[phase_unit]
    phase_axis.set_ylim(float(phase_ticks[0]), float(phase_ticks[-1]))
    phase_axis.set_yticks(phase_ticks)
    phase_axis.set_yticklabels(phase_tick_labels)


def _plot_label(name: str, waveform: Waveform, labels: Mapping[str, str] | None) -> str:
    if labels and name in labels:
        return labels[name]
    return waveform.display_name or waveform.name
