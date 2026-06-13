"""Typed simulation result objects."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeAlias

import numpy as np

from monata.corner import coerce_operating_corner
from monata.measure.result import MeasureSet
from monata.sim import vector_names as _vector_names
from monata.sim import result_ops as _result_ops
from monata.sim._frozen import frozen_array, frozen_mapping
from monata.sim._vector_identity import (
    device_parameter_from_vector,
    is_safe_attribute_name,
    safe_lookup_name,
    simple_vector_inner,
)
from monata.sim.waveform import Waveform as Waveform, _abscissa_unit

if TYPE_CHECKING:
    from monata.measure.freq_domain import BodeTrace
    from monata.units import UnitArray

ResultPayload: TypeAlias = Mapping[str, Any]
WaveformMap: TypeAlias = Mapping[str, Any]


class WaveformNotFoundError(KeyError):
    """Raised when an analysis result cannot resolve a waveform name."""


class SimStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"

    @classmethod
    def coerce(cls, value: "SimStatus | str") -> "SimStatus":
        try:
            return value if isinstance(value, cls) else cls(str(value))
        except ValueError as exc:
            known = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown simulation status {value!r}; expected one of: {known}") from exc


@dataclass(frozen=True, init=False)
class SimResult:
    """Backend-neutral outcome of a simulation task."""

    status: SimStatus
    waveforms: WaveformMap
    sweep_var: np.ndarray | None
    corner: Any
    metadata: ResultPayload
    error_message: str | None
    _analysis_result: AnalysisResult | None
    _measures: MeasureSet
    _summaries: ResultPayload

    def __init__(
        self,
        status: SimStatus | str,
        waveforms: WaveformMap,
        sweep_var: np.ndarray | None,
        corner: Any,
        metadata: ResultPayload | None = None,
        error_message: str | None = None,
        analysis_result: AnalysisResult | None = None,
        measures: MeasureSet | ResultPayload | None = None,
        summaries: ResultPayload | None = None,
    ) -> None:
        object.__setattr__(self, "status", SimStatus.coerce(status))
        object.__setattr__(self, "waveforms", _read_only_mapping(waveforms))
        object.__setattr__(self, "sweep_var", None if sweep_var is None else frozen_array(sweep_var))
        object.__setattr__(self, "corner", coerce_operating_corner(corner))
        object.__setattr__(self, "metadata", _read_only_mapping(metadata))
        object.__setattr__(self, "error_message", error_message)
        object.__setattr__(self, "_analysis_result", analysis_result)
        object.__setattr__(
            self,
            "_measures",
            measures if isinstance(measures, MeasureSet) else MeasureSet(measures),
        )
        object.__setattr__(self, "_summaries", _read_only_mapping(summaries))

    @property
    def analysis_result(self) -> AnalysisResult | None:
        if self.status != SimStatus.OK:
            return None
        if self._analysis_result is None:
            object.__setattr__(
                self,
                "_analysis_result",
                analysis_result_from_arrays(
                    self.waveforms,
                    self.sweep_var,
                    self.metadata,
                ),
            )
        return self._analysis_result

    def waveform(self, name: str) -> Waveform:
        analysis = self.analysis_result
        if analysis is None:
            raise WaveformNotFoundError(
                f"simulation result has no successful analysis result: status={self.status}"
            )
        return analysis.waveform(name)

    def select_waveforms(
        self,
        *names: str,
        missing: Literal["raise", "ignore"] = "raise",
    ) -> dict[str, Waveform]:
        analysis = self.analysis_result
        if analysis is None:
            if missing == "ignore":
                return {}
            raise WaveformNotFoundError(
                f"simulation result has no successful analysis result: status={self.status}"
            )
        return analysis.select_waveforms(*names, missing=missing)

    @property
    def measures(self) -> MeasureSet:
        return self._measures

    @property
    def summaries(self) -> ResultPayload:
        return self._summaries

    def with_summary(self, name: str, summary: Any) -> "SimResult":
        summaries = dict(self.summaries)
        summaries[str(name)] = summary
        return self.with_summaries(summaries)

    def with_summaries(self, summaries: ResultPayload) -> "SimResult":
        return SimResult(
            status=self.status,
            waveforms=self.waveforms,
            sweep_var=self.sweep_var,
            corner=self.corner,
            metadata=self.metadata,
            error_message=self.error_message,
            analysis_result=self._analysis_result,
            measures=self.measures,
            summaries=summaries,
        )


@dataclass(frozen=True)
class AnalysisResult:
    """Typed view of all vectors from one simulator analysis."""

    analysis: str | None
    waveforms: Mapping[str, Waveform]
    abscissa: Waveform | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        waveforms = dict(self.waveforms)
        if self.abscissa is not None:
            waveforms = {
                name: _waveform_with_abscissa_data(waveform, self.abscissa)
                for name, waveform in waveforms.items()
            }
        object.__setattr__(self, "waveforms", frozen_mapping(waveforms))
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    def waveform(self, name: str) -> Waveform:
        for key in _lookup_candidates(name):
            if key in self.waveforms:
                return self.waveforms[key]
        for waveform in self.waveforms.values():
            names = {
                waveform.name,
                waveform.display_name,
                waveform.normalized_name,
                waveform.raw_vector_name,
                waveform.source_vector,
            }
            if any(candidate in names for candidate in _lookup_candidates(name)):
                return waveform
        available = ", ".join(sorted(self.waveforms))
        normalized = safe_lookup_name(name)
        raise WaveformNotFoundError(
            f"waveform not found: {name}; normalized candidate: {normalized}; available: {available}"
        )

    def select_waveforms(
        self,
        *names: str,
        missing: Literal["raise", "ignore"] = "raise",
    ) -> dict[str, Waveform]:
        """Return requested waveforms in request order.

        Keys are the caller's requested names, while values are resolved through
        the same alias-aware lookup as ``waveform``.
        """

        if missing not in {"raise", "ignore"}:
            raise ValueError("missing must be 'raise' or 'ignore'")
        selected: dict[str, Waveform] = {}
        for name in names:
            key = str(name)
            try:
                selected[key] = self.waveform(key)
            except WaveformNotFoundError:
                if missing == "raise":
                    raise
        return selected

    def to_arrays(
        self,
        *names: str,
        include_abscissa: bool = True,
        missing: Literal["raise", "ignore"] = "raise",
        copy: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return abscissa and waveform data arrays in a stable column order."""

        selected = self.select_waveforms(*names, missing=missing) if names else dict(self.waveforms)
        columns: dict[str, np.ndarray] = {}
        abscissa = self.abscissa or _shared_bound_abscissa(selected.values())
        if include_abscissa and abscissa is not None:
            abscissa_name = abscissa.name
            if abscissa_name in selected:
                raise ValueError(f"abscissa column conflicts with waveform column: {abscissa_name}")
            columns[abscissa_name] = abscissa.as_array(copy=copy)
        for name, waveform in selected.items():
            columns[name] = waveform.as_array(copy=copy)
        return columns

    def to_unit_arrays(
        self,
        *names: str,
        include_abscissa: bool = True,
        missing: Literal["raise", "ignore"] = "raise",
        copy: bool = False,
    ) -> dict[str, UnitArray]:
        """Return abscissa and waveform data as unit-bearing arrays in stable column order."""

        selected = self.select_waveforms(*names, missing=missing) if names else dict(self.waveforms)
        columns: dict[str, UnitArray] = {}
        abscissa = self.abscissa or _shared_bound_abscissa(selected.values())
        if include_abscissa and abscissa is not None:
            abscissa_name = abscissa.name
            if abscissa_name in selected:
                raise ValueError(f"abscissa column conflicts with waveform column: {abscissa_name}")
            columns[abscissa_name] = _waveform_unit_array(abscissa, copy=copy)
        for name, waveform in selected.items():
            columns[name] = _waveform_unit_array(waveform, copy=copy)
        return columns

    def bode_trace(
        self,
        name: str,
        *,
        phase_unit: Literal["deg", "rad"] = "deg",
        unwrap_phase: bool = False,
    ) -> BodeTrace:
        """Return a plotting-neutral Bode trace for a complex waveform."""

        from monata.measure.freq_domain import bode_trace

        waveform = self.waveform(name)
        return bode_trace(
            _bode_frequency_abscissa(self, waveform),
            waveform.data,
            phase_unit=phase_unit,
            unwrap_phase=unwrap_phase,
        )

    def __len__(self) -> int:
        return len(self.waveforms)

    def __iter__(self) -> Iterator[str]:
        return iter(self.waveforms)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self.waveform(name)
        except WaveformNotFoundError:
            return False
        return True

    def keys(self) -> KeysView[str]:
        return self.waveforms.keys()

    def values(self) -> ValuesView[Waveform]:
        return self.waveforms.values()

    def items(self) -> ItemsView[str, Waveform]:
        return self.waveforms.items()

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self.waveform(name)
        except WaveformNotFoundError:
            return default

    def __getitem__(self, name: str) -> Waveform:
        return self.waveform(name)

    def __getattr__(self, name: str) -> Waveform:
        if is_safe_attribute_name(name):
            try:
                return self.waveform(name)
            except WaveformNotFoundError:
                pass
        raise AttributeError(name)

    @property
    def abscissa_data(self) -> np.ndarray:
        return self._required_abscissa().data

    @property
    def time(self) -> np.ndarray:
        return self._required_abscissa_named("time").data

    @property
    def frequency(self) -> np.ndarray:
        return self._required_abscissa_named("frequency").data

    @property
    def sweep(self) -> np.ndarray:
        return self._required_abscissa_named("sweep").data

    @property
    def voltages(self) -> dict[str, Waveform]:
        return self._by_quantity_or_kind("voltage", {"node_voltage", "differential_voltage"})

    @property
    def node_voltages(self) -> dict[str, Waveform]:
        return self._by_kind("node_voltage")

    @property
    def node_voltages_by_node(self) -> dict[str, Waveform]:
        """Return node voltage waveforms keyed by circuit node name."""

        return self._by_entity_name("node_voltage")

    @property
    def differential_voltages(self) -> dict[str, Waveform]:
        return self._by_kind("differential_voltage")

    @property
    def currents(self) -> dict[str, Waveform]:
        return self._by_quantity_or_kind("current", {"branch_current", "node_current"})

    @property
    def branch_currents(self) -> dict[str, Waveform]:
        return self._by_kind("branch_current")

    @property
    def branch_currents_by_element(self) -> dict[str, Waveform]:
        """Return branch current waveforms keyed by source or element name."""

        return self._by_entity_name("branch_current")

    @property
    def node_currents(self) -> dict[str, Waveform]:
        return self._by_kind("node_current")

    @property
    def element_parameters(self) -> dict[str, Waveform]:
        return self._by_kind("element_parameter")

    @property
    def device_parameters(self) -> dict[str, Waveform]:
        """Return device parameter waveforms such as ``@m1[gm]``."""

        return {
            name: waveform
            for name, waveform in self.waveforms.items()
            if _device_parameter_for_waveform(waveform) is not None
        }

    @property
    def device_parameters_by_element(self) -> dict[str, dict[str, Waveform]]:
        """Return device parameter waveforms grouped by element and parameter name."""

        grouped: dict[str, dict[str, Waveform]] = {}
        for waveform in self.waveforms.values():
            identity = _device_parameter_for_waveform(waveform)
            if identity is None:
                continue
            element, parameter = identity
            grouped.setdefault(element, {})[parameter] = waveform
        return grouped

    @property
    def internal_parameters(self) -> dict[str, Waveform]:
        return self._by_kind("internal_parameter")

    @property
    def noise(self) -> dict[str, Waveform]:
        return self._by_quantity_or_kind("noise", {"noise_spectrum", "noise_total"})

    @property
    def noise_spectra(self) -> dict[str, Waveform]:
        return self._by_kind("noise_spectrum")

    @property
    def noise_totals(self) -> dict[str, Waveform]:
        return self._by_kind("noise_total")

    @property
    def ac_components(self) -> dict[str, Waveform]:
        return self._by_kind("ac_component")

    @property
    def sensitivities(self) -> dict[str, Waveform]:
        return self._by_kind("sensitivity")

    @property
    def poles(self) -> dict[str, Waveform]:
        return self._by_kind("pole")

    @property
    def zeros(self) -> dict[str, Waveform]:
        return self._by_kind("zero")

    @property
    def pole_zero(self) -> dict[str, Waveform]:
        return self._by_kind("pole", "zero")

    @property
    def distortion(self) -> dict[str, Waveform]:
        return self._by_kind("distortion")

    @property
    def transfer_functions(self) -> dict[str, Waveform]:
        return self._by_kind("transfer_function")

    @property
    def fourier_components(self) -> dict[str, Waveform]:
        return self._by_kind("fourier_component")

    @property
    def expressions(self) -> dict[str, Waveform]:
        return self._by_kind("expression")

    def derivative(self, name: str, result_name: str | None = None, *, edge_order: Literal[1, 2] = 1) -> Waveform:
        """Return a waveform derivative using this result's abscissa."""

        return _result_ops.derivative(self, name, result_name, edge_order=edge_order)

    def integral(self, name: str, result_name: str | None = None, *, initial: Any = 0) -> Waveform:
        """Return a cumulative waveform integral using this result's abscissa."""

        return _result_ops.integral(self, name, result_name, initial=initial)

    def resample(self, name: str, target_abscissa: Any, result_name: str | None = None) -> Waveform:
        """Return a waveform interpolated onto a new one-dimensional abscissa."""

        return _result_ops.resample(self, name, target_abscissa, result_name)

    def window(self, name: str, start: Any = None, stop: Any = None, result_name: str | None = None) -> Waveform:
        """Return a waveform subset using this result's abscissa as a closed interval."""

        return _result_ops.window(self, name, start, stop, result_name)

    def windowed(
        self,
        start: Any = None,
        stop: Any = None,
        *names: str,
        missing: Literal["raise", "ignore"] = "raise",
    ) -> AnalysisResult:
        """Return a new analysis result cropped to a shared abscissa interval."""

        return _result_ops.windowed(self, start, stop, *names, missing=missing)

    def sample_at(
        self,
        name: str,
        target_abscissa: Any,
        *,
        left: Any = None,
        right: Any = None,
        with_unit: bool = False,
    ) -> Any:
        """Return linearly interpolated waveform value(s) using this result's abscissa."""

        return _result_ops.sample_at(
            self,
            name,
            target_abscissa,
            left=left,
            right=right,
            with_unit=with_unit,
        )

    def _by_quantity_or_kind(self, quantity: str, kinds: set[str]) -> dict[str, Waveform]:
        return {
            name: waveform
            for name, waveform in self.waveforms.items()
            if waveform.quantity == quantity or waveform.vector_kind in kinds
        }

    def _by_kind(self, *kinds: str) -> dict[str, Waveform]:
        return {name: waveform for name, waveform in self.waveforms.items() if waveform.vector_kind in kinds}

    def _by_entity_name(self, kind: str) -> dict[str, Waveform]:
        return {
            _entity_name_for_waveform(waveform, kind): waveform
            for waveform in self.waveforms.values()
            if waveform.vector_kind == kind
        }

    def _required_abscissa(self) -> Waveform:
        if self.abscissa is None:
            raise ValueError("analysis result has no abscissa waveform")
        return self.abscissa

    def _required_abscissa_named(self, name: str) -> Waveform:
        abscissa = self._required_abscissa()
        if abscissa.name != name:
            raise ValueError(f"analysis result abscissa is {abscissa.name!r}, not {name!r}")
        return abscissa


def analysis_result_from_arrays(
    waveforms: Mapping[str, Any],
    sweep_var: Any | None,
    metadata: Mapping[str, Any] | None = None,
    *,
    analysis: str | None = None,
    source: str | None = None,
    source_vectors: Mapping[str, str] | None = None,
) -> AnalysisResult:
    """Build typed result objects from waveform arrays."""

    metadata_dict = dict(metadata or {})
    analysis_name = analysis or metadata_dict.get("analysis")
    source_name = source or metadata_dict.get("extraction")
    vector_kinds = dict(metadata_dict.get("vector_kinds", {}))
    vector_quantities = dict(metadata_dict.get("vector_quantities", {}))
    vector_units = dict(metadata_dict.get("vector_units", {}))
    vector_raw_names = dict(metadata_dict.get("vector_raw_names", {}))
    vector_metadata = dict(metadata_dict.get("vector_metadata", {}))
    abscissa = _abscissa_waveform(
        sweep_var,
        str(analysis_name) if analysis_name else None,
        metadata_dict,
    )
    typed_waveforms: dict[str, Waveform] = {}
    for name, values in waveforms.items():
        source_vector = vector_raw_names.get(str(name), (source_vectors or {}).get(str(name)))
        quantity_name = source_vector or str(name)
        vector_name = _vector_names.normalize_vector_name(quantity_name, display_name=str(name))
        vector_kind = vector_kinds.get(str(name), vector_name.vector_kind)
        quantity = vector_quantities.get(str(name), vector_name.quantity)
        unit_name = vector_units.get(str(name), _unit_for_name(vector_name.raw_vector_name))
        typed_waveforms[str(name)] = Waveform(
            name=str(name),
            data=np.asarray(values),
            unit=unit_name,
            quantity=quantity,
            display_name=vector_name.display_name,
            normalized_name=vector_name.normalized_name,
            vector_kind=vector_kind,
            source_vector=source_vector,
            raw_vector_name=vector_name.raw_vector_name,
            abscissa=abscissa.name if abscissa is not None else None,
            abscissa_data=abscissa.data if abscissa is not None else None,
            metadata=dict(vector_metadata.get(str(name), {})),
            analysis=str(analysis_name) if analysis_name else None,
            source=str(source_name) if source_name else None,
            extraction=str(source_name) if source_name else None,
        )
    return AnalysisResult(
        analysis=str(analysis_name) if analysis_name else None,
        waveforms=typed_waveforms,
        abscissa=abscissa,
        metadata=metadata_dict,
        source=str(source_name) if source_name else None,
    )


def _abscissa_waveform(
    sweep_var: Any | None,
    analysis: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> Waveform | None:
    if sweep_var is None:
        return None
    name = _abscissa_name(analysis, metadata)
    return Waveform(
        name=name,
        data=_abscissa_data(name, sweep_var),
        unit=_abscissa_unit(name),
        quantity=name,
        source_vector=name,
        raw_vector_name=name,
        vector_kind="abscissa",
    )


def _waveform_unit_array(waveform: Waveform, *, copy: bool = False) -> UnitArray:
    unit_array = waveform.unit_array()
    if not copy:
        return unit_array
    from monata.units import UnitArray

    return UnitArray(unit_array.as_array(copy=True), unit_array.unit)


def _bode_frequency_abscissa(result: AnalysisResult, waveform: Waveform) -> Any:
    if result.abscissa is not None:
        abscissa = result._required_abscissa_named("frequency")
        return abscissa.unit_array() if abscissa.unit is not None else abscissa.data
    if waveform.abscissa_data is not None and waveform.abscissa_name == "frequency":
        return waveform.abscissa_data
    raise ValueError("Bode trace requires a frequency abscissa")


def _shared_bound_abscissa(waveforms: ValuesView[Waveform]) -> Waveform | None:
    iterator = iter(waveforms)
    try:
        first = next(iterator)
    except StopIteration:
        return None
    if first.abscissa_data is None:
        return None
    name = first.abscissa_name or first.abscissa or "abscissa"
    data = np.asarray(first.abscissa_data)
    for waveform in iterator:
        if waveform.abscissa_data is None:
            return None
        waveform_name = waveform.abscissa_name or waveform.abscissa or "abscissa"
        if waveform_name != name or not np.array_equal(np.asarray(waveform.abscissa_data), data):
            return None
    return Waveform(
        name=name,
        data=data,
        unit=_abscissa_unit(name),
        quantity=name,
        source_vector=name,
        raw_vector_name=name,
        vector_kind="abscissa",
    )


def _abscissa_name(analysis: str | None, metadata: Mapping[str, Any] | None = None) -> str:
    analysis_name = str(analysis).lower() if analysis else ""
    if analysis_name == "sens" and _is_ac_sensitivity_metadata(metadata):
        return "frequency"
    return {
        "ac": "frequency",
        "disto": "frequency",
        "four": "frequency",
        "noise": "frequency",
        "tran": "time",
        "dc": "sweep",
    }.get(analysis_name, "sweep")


def _is_ac_sensitivity_metadata(metadata: Mapping[str, Any] | None) -> bool:
    if metadata is None:
        return False
    return metadata.get("start") is not None and metadata.get("stop") is not None and metadata.get("points") is not None


def _abscissa_data(name: str, values: Any) -> np.ndarray:
    data = np.asarray(values)
    if name in {"frequency", "time"} and np.iscomplexobj(data):
        imaginary = np.imag(data)
        if np.allclose(imaginary, 0.0):
            return np.real(data)
    return data


def _waveform_with_abscissa_data(waveform: Waveform, abscissa: Waveform) -> Waveform:
    if waveform.abscissa_data is not None:
        return waveform
    if waveform.abscissa_name is not None and waveform.abscissa_name != abscissa.name:
        return waveform
    return replace(waveform, abscissa=abscissa.name, abscissa_data=abscissa.data)


def _entity_name_for_waveform(waveform: Waveform, kind: str) -> str:
    candidates = (
        waveform.source_vector,
        waveform.raw_vector_name,
        waveform.display_name,
        waveform.name,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        entity = _entity_name_from_vector(candidate, kind)
        if entity is not None:
            return entity
    return str(waveform.display_name or waveform.name)


def _entity_name_from_vector(name: str, kind: str) -> str | None:
    text = str(name).strip()
    lower = text.lower()
    voltage_inner = simple_vector_inner(text, "v")
    if kind == "node_voltage" and voltage_inner is not None:
        inner = voltage_inner.strip()
        if inner and "," not in inner:
            return inner
    current_inner = simple_vector_inner(text, "i")
    if kind == "branch_current" and current_inner is not None:
        inner = current_inner.strip()
        if inner:
            return inner
    if kind == "branch_current" and lower.endswith("#branch"):
        inner = text[:-7].strip()
        if inner:
            return inner
    return None


def _device_parameter_for_waveform(waveform: Waveform) -> tuple[str, str] | None:
    candidates = (
        waveform.source_vector,
        waveform.raw_vector_name,
        waveform.display_name,
        waveform.name,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        parameter = device_parameter_from_vector(candidate)
        if parameter is not None:
            return parameter
    return None


def _lookup_candidates(name: str) -> tuple[str, ...]:
    raw = str(name)
    text = raw.strip()
    vector = _vector_names.normalize_vector_name(text)
    candidates = (
        raw,
        text,
        text.lower(),
        safe_lookup_name(text),
        vector.display_name,
        vector.display_name.lower(),
        vector.normalized_name,
        vector.raw_vector_name,
        vector.raw_vector_name.lower(),
    )
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _unit_for_name(name: str) -> str | None:
    quantity = _vector_names.normalize_vector_name(name).quantity
    if quantity == "voltage":
        return "V"
    if quantity == "current":
        return "A"
    return None


def _read_only_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return frozen_mapping(values)


__all__ = [
    "AnalysisResult",
    "ResultPayload",
    "SimResult",
    "SimStatus",
    "Waveform",
    "WaveformMap",
    "WaveformNotFoundError",
    "analysis_result_from_arrays",
]
