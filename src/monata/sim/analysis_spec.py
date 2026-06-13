"""Lazy analysis descriptions as immutable API contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


_VARIATIONS = ("dec", "oct", "lin")
_POLE_ZERO_TRANSFERS = ("vol", "cur")
_POLE_ZERO_MODES = ("pol", "zer", "pz")
_CONTROL_CHARS = "\r\n;"
_ANALYSIS_NAMES = {
    "AC": "ac",
    "DC": "dc",
    "Tran": "tran",
    "OP": "op",
    "Noise": "noise",
    "Sensitivity": "sensitivity",
    "PoleZero": "pole-zero",
    "Distortion": "distortion",
    "TransferFunction": "transfer-function",
    "Fourier": "fourier",
}


@dataclass(frozen=True)
class AnalysisSpec:
    """Base class for concrete simulation analysis specifications."""


@dataclass(frozen=True)
class ACSpec(AnalysisSpec):
    start: float
    stop: float
    points: int
    variation: str = "dec"

    def __post_init__(self) -> None:
        _validate_positive_numeric(self.start, "ac start")
        _validate_positive_numeric(self.stop, "ac stop")
        _validate_greater_than(self.stop, self.start, "ac stop", "ac start")
        _validate_positive_integer(self.points, "ac points")
        _normalize_field(self, "variation", _choice(self.variation, "AC variation", _VARIATIONS))


@dataclass(frozen=True)
class TranSpec(AnalysisSpec):
    stop: float
    step: float | None = None
    start: float = 0
    max_step: float | None = None
    uic: bool = False

    def __post_init__(self) -> None:
        _validate_nonnegative_numeric(self.start, "tran start")
        _validate_positive_numeric(self.stop, "tran stop")
        _validate_greater_than(self.stop, self.start, "tran stop", "tran start")
        if self.step is not None:
            _validate_positive_numeric(self.step, "tran step")
        if self.max_step is not None:
            _validate_positive_numeric(self.max_step, "tran max_step")
        _normalize_field(self, "uic", bool(self.uic))


@dataclass(frozen=True)
class DCSweep:
    source: str
    start: float
    stop: float
    step: float

    def __post_init__(self) -> None:
        _validate_text(self.source, "dc source")
        _validate_numeric(self.start, "dc start")
        _validate_numeric(self.stop, "dc stop")
        _validate_nonzero_numeric(self.step, "dc step")


@dataclass(frozen=True)
class DCSpec(AnalysisSpec):
    source: str
    start: float
    stop: float
    step: float
    secondary: DCSweep | None = None

    def __post_init__(self) -> None:
        _validate_text(self.source, "dc source")
        _validate_numeric(self.start, "dc start")
        _validate_numeric(self.stop, "dc stop")
        _validate_nonzero_numeric(self.step, "dc step")
        if self.secondary is not None and not isinstance(self.secondary, DCSweep):
            raise ValueError("invalid dc secondary: expected DCSweep")


@dataclass(frozen=True)
class OPSpec(AnalysisSpec):
    pass


@dataclass(frozen=True)
class NoiseSpec(AnalysisSpec):
    output_node: str
    input_source: str
    start: float
    stop: float
    points: int
    reference_node: str = "0"
    variation: str = "dec"
    points_per_summary: int | None = None

    def __post_init__(self) -> None:
        _validate_text(self.output_node, "noise output")
        _validate_text(self.input_source, "noise input_source")
        _validate_text(self.reference_node, "noise reference")
        _validate_positive_numeric(self.start, "noise start")
        _validate_positive_numeric(self.stop, "noise stop")
        _validate_greater_than(self.stop, self.start, "noise stop", "noise start")
        _validate_positive_integer(self.points, "noise points")
        _normalize_field(self, "variation", _choice(self.variation, "noise variation", _VARIATIONS))
        if self.points_per_summary is not None:
            _validate_positive_integer(self.points_per_summary, "noise points_per_summary")


@dataclass(frozen=True)
class SensitivitySpec(AnalysisSpec):
    output: str
    start: float | None = None
    stop: float | None = None
    points: int | None = None
    variation: str = "dec"

    def __post_init__(self) -> None:
        _validate_text(self.output, "sensitivity output")
        _normalize_field(self, "variation", _choice(self.variation, "sensitivity variation", _VARIATIONS))
        sweep_fields = (self.start, self.stop, self.points)
        if any(value is not None for value in sweep_fields) and not all(value is not None for value in sweep_fields):
            raise ValueError("SensitivitySpec AC sweep requires start, stop, and points together")
        if self.start is not None and self.stop is not None and self.points is not None:
            _validate_positive_numeric(self.start, "sensitivity start")
            _validate_positive_numeric(self.stop, "sensitivity stop")
            _validate_greater_than(self.stop, self.start, "sensitivity stop", "sensitivity start")
            _validate_positive_integer(self.points, "sensitivity points")


@dataclass(frozen=True)
class PoleZeroSpec(AnalysisSpec):
    input_pos: str
    input_neg: str
    output_pos: str
    output_neg: str
    transfer: str = "vol"
    mode: str = "pz"

    def __post_init__(self) -> None:
        _validate_text(self.input_pos, "pole-zero input_pos")
        _validate_text(self.input_neg, "pole-zero input_neg")
        _validate_text(self.output_pos, "pole-zero output_pos")
        _validate_text(self.output_neg, "pole-zero output_neg")
        _normalize_field(self, "transfer", _choice(self.transfer, "pole-zero transfer", _POLE_ZERO_TRANSFERS))
        _normalize_field(self, "mode", _choice(self.mode, "pole-zero mode", _POLE_ZERO_MODES))


@dataclass(frozen=True)
class DistortionSpec(AnalysisSpec):
    start: float
    stop: float
    points: int
    variation: str = "dec"
    f2overf1: float | None = None

    def __post_init__(self) -> None:
        _validate_positive_numeric(self.start, "distortion start")
        _validate_positive_numeric(self.stop, "distortion stop")
        _validate_greater_than(self.stop, self.start, "distortion stop", "distortion start")
        _validate_positive_integer(self.points, "distortion points")
        _normalize_field(self, "variation", _choice(self.variation, "distortion variation", _VARIATIONS))
        if self.f2overf1 is not None:
            _validate_positive_numeric(self.f2overf1, "distortion f2overf1")


@dataclass(frozen=True)
class TransferFunctionSpec(AnalysisSpec):
    output: str
    input_source: str

    def __post_init__(self) -> None:
        _validate_text(self.output, "tf output")
        _validate_text(self.input_source, "tf input_source")


@dataclass(frozen=True)
class FourierSpec(AnalysisSpec):
    frequency: float
    output: str
    stop: float
    step: float | None = None
    start: float = 0

    def __post_init__(self) -> None:
        _validate_positive_numeric(self.frequency, "fourier frequency")
        _validate_text(self.output, "fourier output")
        _validate_nonnegative_numeric(self.start, "fourier start")
        _validate_positive_numeric(self.stop, "fourier stop")
        _validate_greater_than(self.stop, self.start, "fourier stop", "fourier start")
        if self.step is not None:
            _validate_positive_numeric(self.step, "fourier step")


def analysis_name(analysis_spec: object) -> str:
    """Return the canonical generic analysis name for an analysis spec."""

    name = type(analysis_spec).__name__
    if name.endswith("Spec"):
        name = name[:-4]
    return _ANALYSIS_NAMES.get(name, name.lower())


def _normalize_field(instance: AnalysisSpec, field_name: str, value: object) -> None:
    object.__setattr__(instance, field_name, value)


def _choice(value: str, label: str, choices: Iterable[str]) -> str:
    _validate_text(value, label)
    text = value.lower()
    allowed = tuple(choices)
    if text not in allowed:
        expected = ", ".join(allowed)
        raise ValueError(f"invalid {label}: {value}; expected one of: {expected}")
    return text


def _validate_text(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"invalid {label}: expected string")
    if not value.strip():
        raise ValueError(f"invalid {label}: value is required")
    if any(char in value for char in _CONTROL_CHARS):
        raise ValueError(f"invalid {label}: control characters are not allowed")


def _validate_numeric(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid {label}: expected numeric scalar")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: expected numeric scalar") from exc
    if not math.isfinite(number):
        raise ValueError(f"invalid {label}: expected finite numeric scalar")
    _validate_scalar_text(value, label)
    return number


def _validate_positive_numeric(value: Any, label: str) -> float:
    number = _validate_numeric(value, label)
    if number <= 0:
        raise ValueError(f"invalid {label}: expected positive numeric scalar")
    return number


def _validate_nonnegative_numeric(value: Any, label: str) -> float:
    number = _validate_numeric(value, label)
    if number < 0:
        raise ValueError(f"invalid {label}: expected non-negative numeric scalar")
    return number


def _validate_nonzero_numeric(value: Any, label: str) -> float:
    number = _validate_numeric(value, label)
    if number == 0:
        raise ValueError(f"invalid {label}: expected non-zero numeric scalar")
    return number


def _validate_greater_than(value: Any, lower_bound: Any, label: str, lower_label: str) -> None:
    if float(value) <= float(lower_bound):
        raise ValueError(f"invalid {label}: expected greater than {lower_label}")


def _validate_positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid {label}: expected positive integer")
    if value <= 0:
        raise ValueError(f"invalid {label}: expected positive integer")


def _validate_scalar_text(value: Any, label: str) -> None:
    text = str(value)
    if any(char in text for char in _CONTROL_CHARS):
        raise ValueError(f"invalid {label}: control characters are not allowed")


__all__ = [
    "ACSpec",
    "AnalysisSpec",
    "DCSweep",
    "DCSpec",
    "DistortionSpec",
    "FourierSpec",
    "NoiseSpec",
    "OPSpec",
    "PoleZeroSpec",
    "SensitivitySpec",
    "TranSpec",
    "TransferFunctionSpec",
    "analysis_name",
]
