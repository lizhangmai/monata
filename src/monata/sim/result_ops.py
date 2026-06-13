"""Numeric operations over simulation result containers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from monata.sim.results import AnalysisResult
    from monata.sim.waveform import Waveform

__all__ = [
    "derivative",
    "integral",
    "resample",
    "sample_at",
    "window",
    "windowed",
]


def derivative(
    result: AnalysisResult,
    name: str,
    result_name: str | None = None,
    *,
    edge_order: Literal[1, 2] = 1,
) -> Waveform:
    """Return a waveform derivative using an analysis result's abscissa."""

    waveform = result.waveform(name)
    return waveform.derivative(_waveform_source_abscissa(result, waveform), result_name, edge_order=edge_order)


def integral(
    result: AnalysisResult,
    name: str,
    result_name: str | None = None,
    *,
    initial: Any = 0,
) -> Waveform:
    """Return a cumulative waveform integral using an analysis result's abscissa."""

    waveform = result.waveform(name)
    return waveform.integral(_waveform_source_abscissa(result, waveform), result_name, initial=initial)


def resample(
    result: AnalysisResult,
    name: str,
    target_abscissa: Any,
    result_name: str | None = None,
) -> Waveform:
    """Return a waveform interpolated onto a new one-dimensional abscissa."""

    waveform = result.waveform(name)
    return waveform.resample(
        target_abscissa,
        source_abscissa=_waveform_source_abscissa(result, waveform),
        name=result_name,
    )


def window(
    result: AnalysisResult,
    name: str,
    start: Any = None,
    stop: Any = None,
    result_name: str | None = None,
) -> Waveform:
    """Return a waveform subset using an analysis result's abscissa as a closed interval."""

    waveform = result.waveform(name)
    return waveform.window(
        start,
        stop,
        source_abscissa=_waveform_source_abscissa(result, waveform),
        name=result_name,
    )


def windowed(
    result: AnalysisResult,
    start: Any = None,
    stop: Any = None,
    *names: str,
    missing: Literal["raise", "ignore"] = "raise",
) -> AnalysisResult:
    """Return a new analysis result cropped to a shared abscissa interval."""

    from monata.sim.results import AnalysisResult

    abscissa = _required_abscissa(result)
    windowed_abscissa = abscissa.window(start, stop, source_abscissa=abscissa)
    selected = result.select_waveforms(*names, missing=missing) if names else dict(result.waveforms)
    waveforms = {
        key: waveform.window(
            start,
            stop,
            source_abscissa=_waveform_source_abscissa(result, waveform),
        )
        for key, waveform in selected.items()
    }
    return AnalysisResult(
        analysis=result.analysis,
        waveforms=waveforms,
        abscissa=windowed_abscissa,
        metadata=dict(result.metadata),
        source=result.source,
    )


def sample_at(
    result: AnalysisResult,
    name: str,
    target_abscissa: Any,
    *,
    left: Any = None,
    right: Any = None,
    with_unit: bool = False,
) -> Any:
    """Return linearly interpolated waveform value(s) using an analysis result's abscissa."""

    waveform = result.waveform(name)
    return waveform.sample_at(
        target_abscissa,
        source_abscissa=_waveform_source_abscissa(result, waveform),
        left=left,
        right=right,
        with_unit=with_unit,
    )


def _required_abscissa(result: AnalysisResult) -> Waveform:
    if result.abscissa is None:
        raise ValueError("analysis result has no abscissa waveform")
    return result.abscissa


def _waveform_source_abscissa(result: AnalysisResult, waveform: Waveform) -> Waveform | None:
    return None if waveform.abscissa_data is not None else _required_abscissa(result)
