"""Typed waveform value object and numeric helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np

from monata.sim import vector_names as _vector_names
from monata.sim._frozen import frozen_array, frozen_mapping
from monata.sim._vector_identity import safe_lookup_name

if TYPE_CHECKING:
    from monata.units import UnitArray


_PRESERVE_METADATA = object()
_WAVEFORM_COMPARISON_UFUNCS = {
    np.equal,
    np.not_equal,
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
}
_WAVEFORM_UNARY_PRESERVE_UFUNCS = {
    np.negative,
    np.positive,
    np.absolute,
    np.fabs,
    np.rint,
    np.floor,
    np.ceil,
    np.trunc,
    np.conjugate,
}
_WAVEFORM_BINARY_PRESERVE_UFUNCS = {
    np.add,
    np.subtract,
    np.multiply,
    np.divide,
    np.true_divide,
}


@dataclass(frozen=True)
class Waveform:
    """A typed view of a simulator vector."""

    name: str
    data: np.ndarray
    unit: Any = None
    quantity: str | None = None
    title: str | None = None
    source_vector: str | None = None
    abscissa: str | None = None
    abscissa_data: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    display_name: str | None = None
    normalized_name: str | None = None
    raw_vector_name: str | None = None
    vector_kind: str = "unknown"
    analysis: str | None = None
    source: str | None = None
    extraction: str | None = None
    abscissa_name: str | None = None
    plot_name: str | None = None

    @classmethod
    def from_array(
        cls,
        name: str,
        array: Any,
        title: str | None = None,
        abscissa: Any = None,
        *,
        unit: Any = None,
        quantity: str | None = None,
        abscissa_name: str | None = None,
        **metadata: Any,
    ) -> Waveform:
        """Build a waveform from a plain array and optional abscissa data."""

        resolved_abscissa_name, abscissa_data = _coerce_abscissa_input(abscissa, abscissa_name)
        return cls(
            name=name,
            data=np.asarray(array),
            unit=unit,
            quantity=quantity,
            title=title,
            abscissa=resolved_abscissa_name,
            abscissa_data=abscissa_data,
            metadata=metadata,
        )

    @classmethod
    def from_unit_array(
        cls,
        name: str,
        array: UnitArray,
        title: str | None = None,
        abscissa: Any = None,
        *,
        quantity: str | None = None,
        abscissa_name: str | None = None,
        **metadata: Any,
    ) -> Waveform:
        """Build a waveform from a Monata UnitArray."""

        from monata.units import UnitArray

        if not isinstance(array, UnitArray):
            raise TypeError("from_unit_array expects a monata.units.UnitArray")
        return cls.from_array(
            name,
            array.values,
            title=title,
            abscissa=abscissa,
            unit=array.unit.symbol,
            quantity=quantity,
            abscissa_name=abscissa_name,
            **metadata,
        )

    def __post_init__(self) -> None:
        data = frozen_array(self.data)
        abscissa_data = _optional_abscissa_data(self.abscissa_data, data)
        object.__setattr__(self, "data", data)
        object.__setattr__(
            self,
            "abscissa_data",
            None if abscissa_data is None else frozen_array(abscissa_data),
        )
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))
        raw_vector_name = self.raw_vector_name or self.source_vector or self.name
        vector_kind = self.vector_kind if self.vector_kind in _vector_names.VECTOR_KINDS else "unknown"
        object.__setattr__(self, "display_name", self.display_name or self.name)
        object.__setattr__(self, "normalized_name", self.normalized_name or safe_lookup_name(self.name))
        object.__setattr__(self, "raw_vector_name", raw_vector_name)
        object.__setattr__(self, "source_vector", self.source_vector or raw_vector_name)
        object.__setattr__(self, "vector_kind", vector_kind)
        object.__setattr__(self, "abscissa_name", self.abscissa_name or self.abscissa)
        object.__setattr__(self, "plot_name", self.plot_name or self.title)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def size(self) -> int:
        return int(self.data.size)

    def __array__(self, dtype: Any | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is None:
            return np.asarray(self.data, dtype=dtype)
        return np.array(self.data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        array_inputs = tuple(input_.data if isinstance(input_, Waveform) else input_ for input_ in inputs)
        result = ufunc(*array_inputs, **kwargs)
        if _waveform_ufunc_returns_array(ufunc, result):
            return result
        primary = _primary_waveform_input(inputs)
        if primary is None:
            return result
        unit_result = _waveform_unit_array_ufunc(ufunc, inputs)
        if unit_result is not None:
            data, unit = unit_result
            return primary._derived(
                _waveform_ufunc_name(ufunc, primary),
                data,
                unit=unit,
                quantity=_waveform_ufunc_quantity(ufunc, inputs, primary),
            )
        if _waveform_ufunc_preserves_unit(ufunc, inputs, primary):
            return primary._derived(
                _waveform_ufunc_name(ufunc, primary),
                result,
                quantity=_waveform_ufunc_quantity(ufunc, inputs, primary),
            )
        if primary.unit is None:
            return primary._derived(_waveform_ufunc_name(ufunc, primary), result, unit=None, quantity=primary.quantity)
        return result

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key: Any) -> Any:
        value = self.data[key]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return replace(self, data=value, abscissa_data=_sliced_abscissa_data(self.abscissa_data, key, value))
        if isinstance(value, np.generic):
            return value.item()
        return value

    def __str__(self) -> str:
        return str(self.title if self.title is not None else self.name)

    def __repr__(self) -> str:
        return f"{type(self).__name__} {self.name} {self.str_data()}"

    def str_data(self) -> str:
        """Return a compact representation of this waveform's numeric data."""

        return repr(self.as_array())

    def clone(self, **updates: Any) -> Waveform:
        """Return a copy with selected fields replaced."""

        if "title" in updates:
            title = updates["title"]
            updates["title"] = None if title is None else str(title)
            updates.setdefault("plot_name", updates["title"])
        if "plot_name" in updates and updates["plot_name"] is not None:
            updates["plot_name"] = str(updates["plot_name"])
        return replace(self, **updates)

    def with_title(self, title: Any | None) -> Waveform:
        """Return a copy with an updated display title."""

        return self.clone(title=title)

    def as_array(self, *, copy: bool = False) -> np.ndarray:
        """Return this waveform's numeric data as a read-only view or writable copy."""

        return np.array(self.data, copy=copy)

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        """Return the waveform sum, preserving known unit metadata when possible."""

        return _waveform_summary(np.sum, self, args, kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        """Return the waveform mean, preserving known unit metadata when possible."""

        return _waveform_summary(np.mean, self, args, kwargs)

    def min(self, *args: Any, **kwargs: Any) -> Any:
        """Return the waveform minimum, preserving known unit metadata when possible."""

        return _waveform_summary(np.min, self, args, kwargs)

    def max(self, *args: Any, **kwargs: Any) -> Any:
        """Return the waveform maximum, preserving known unit metadata when possible."""

        return _waveform_summary(np.max, self, args, kwargs)

    def std(self, *args: Any, **kwargs: Any) -> Any:
        """Return the waveform standard deviation, preserving known unit metadata when possible."""

        return _waveform_summary(np.std, self, args, kwargs)

    def peak_to_peak(self, *args: Any, **kwargs: Any) -> Any:
        """Return the waveform maximum-minus-minimum span."""

        return _waveform_summary(np.ptp, self, args, kwargs)

    def rms(self, *args: Any, **kwargs: Any) -> Any:
        """Return the root-mean-square value of the waveform."""

        if kwargs.get("out") is not None:
            raise ValueError("Waveform summary operations do not support out")
        value = np.sqrt(np.mean(np.abs(self.data) ** 2, *args, **kwargs))
        return _waveform_quantity_or_value(self, value)

    def magnitude(self, name: str | None = None) -> Waveform:
        return self._derived(name or f"{self.name}.mag", np.abs(self.data), quantity=self.quantity)

    def phase(self, name: str | None = None, *, degrees: bool = False, unwrap: bool = False) -> Waveform:
        data = np.angle(self.data)
        if unwrap:
            data = np.unwrap(data)
        if degrees:
            data = np.degrees(data)
        return self._derived(name or f"{self.name}.phase", data, unit="deg" if degrees else "rad", quantity="phase")

    def db(self, name: str | None = None) -> Waveform:
        with np.errstate(divide="ignore"):
            data = 20 * np.log10(np.abs(self.data))
        return self._derived(name or f"{self.name}.db", data, unit="dB", quantity="gain_db")

    def real(self, name: str | None = None) -> Waveform:
        return self._derived(name or f"{self.name}.real", np.real(self.data), quantity=self.quantity)

    def imaginary(self, name: str | None = None) -> Waveform:
        return self._derived(name or f"{self.name}.imag", np.imag(self.data), quantity=self.quantity)

    def derivative(self, abscissa: Any = None, name: str | None = None, *, edge_order: Literal[1, 2] = 1) -> Waveform:
        """Return the numerical derivative against an explicit or stored abscissa."""

        x_values, x_unit = _waveform_abscissa_values(self, abscissa)
        _validate_matching_abscissa(self.data, x_values)
        data = np.gradient(self.data, x_values, edge_order=edge_order)
        return self._derived(
            name or f"d({self.name})",
            data,
            unit=_derived_unit(self.unit, x_unit, "/"),
            quantity=_derived_quantity(self.quantity, "derivative"),
        )

    def integral(self, abscissa: Any = None, name: str | None = None, *, initial: Any = 0) -> Waveform:
        """Return the cumulative trapezoidal integral against an explicit or stored abscissa."""

        x_values, x_unit = _waveform_abscissa_values(self, abscissa)
        _validate_matching_abscissa(self.data, x_values)
        increments = 0.5 * (self.data[1:] + self.data[:-1]) * np.diff(x_values)
        data = np.empty(self.data.shape, dtype=np.result_type(self.data, x_values, initial, float))
        data[0] = initial
        data[1:] = initial + np.cumsum(increments)
        return self._derived(
            name or f"int({self.name})",
            data,
            unit=_derived_unit(self.unit, x_unit, "*"),
            quantity=_derived_quantity(self.quantity, "integral"),
        )

    def resample(
        self,
        target_abscissa: Any,
        *,
        source_abscissa: Any = None,
        name: str | None = None,
        left: Any = None,
        right: Any = None,
    ) -> Waveform:
        """Return this waveform interpolated onto a new one-dimensional abscissa."""

        source_values, _ = _waveform_abscissa_values(self, source_abscissa)
        target_values, target_name = _target_abscissa_values(target_abscissa, self.abscissa_name or self.abscissa)
        _validate_interpolation_abscissas(
            self.data,
            source_values,
            target_values,
            "resampling",
        )
        data = _interp_waveform_data(self.data, source_values, target_values, left=left, right=right)
        result_name = name or f"{self.name}.resampled"
        return Waveform(
            name=result_name,
            data=data,
            unit=self.unit,
            quantity=self.quantity,
            title=self.title,
            source_vector=self.source_vector,
            abscissa=target_name,
            abscissa_data=target_values,
            metadata={**self.metadata, "derived_from": self.name},
            display_name=result_name,
            raw_vector_name=self.raw_vector_name,
            vector_kind=self.vector_kind,
            analysis=self.analysis,
            source=self.source,
            extraction=self.extraction,
            abscissa_name=target_name,
            plot_name=self.plot_name,
        )

    def sample_at(
        self,
        target_abscissa: Any,
        *,
        source_abscissa: Any = None,
        left: Any = None,
        right: Any = None,
        with_unit: bool = False,
    ) -> Any:
        """Return linearly interpolated waveform value(s) at target abscissa point(s)."""

        source_values, _ = _waveform_abscissa_values(self, source_abscissa)
        target_values, scalar = _sample_abscissa_values(target_abscissa)
        _validate_interpolation_abscissas(
            self.data,
            source_values,
            target_values,
            "sampling",
        )
        data = _interp_waveform_data(self.data, source_values, target_values, left=left, right=right)
        if with_unit:
            return _waveform_quantity_or_value(self, _plain_numpy_result(np.asarray(data)[0]) if scalar else data)
        if scalar:
            return _plain_numpy_result(np.asarray(data)[0])
        return data

    def window(
        self,
        start: Any = None,
        stop: Any = None,
        *,
        source_abscissa: Any = None,
        name: str | None = None,
    ) -> Waveform:
        """Return samples whose abscissa lies inside a closed interval."""

        source_values, source_unit = _waveform_abscissa_values(self, source_abscissa)
        source_values = _real_window_values(source_values, "source abscissa")
        _validate_window_abscissa(self.data, source_values)
        start_value = np.min(source_values) if start is None else _window_bound_value(start, "start", source_unit)
        stop_value = np.max(source_values) if stop is None else _window_bound_value(stop, "stop", source_unit)
        if start_value > stop_value:
            raise ValueError("waveform window start must be less than or equal to stop")
        mask = (source_values >= start_value) & (source_values <= stop_value)
        if not np.any(mask):
            raise ValueError("waveform window contains no samples")
        result_name = str(name) if name is not None else self.name
        return replace(
            self,
            name=result_name,
            data=self.data[mask],
            abscissa=_source_abscissa_name(source_abscissa, self.abscissa_name or self.abscissa),
            abscissa_data=source_values[mask],
            display_name=result_name if name is not None else self.display_name,
            normalized_name=None if name is not None else self.normalized_name,
        )

    def unit_array(self) -> UnitArray:
        """Return this waveform as a Monata UnitArray when unit metadata is available."""

        if self.unit is None:
            raise ValueError(f"waveform {self.name} has no unit metadata")
        from monata.units import UnitArray, quantity

        values = quantity(self.data, str(self.unit))
        if not isinstance(values, UnitArray):
            raise TypeError("waveform array conversion produced a scalar quantity")
        return values

    def to_unit(self, target: str, name: str | None = None) -> Waveform:
        """Return a copy converted to a compatible Monata unit."""

        converted = self.unit_array().to(target)
        return self._derived(name or self.name, converted.values, unit=converted.unit.symbol, quantity=self.quantity)

    def _derived(
        self,
        name: str,
        data: Any,
        *,
        unit: Any = _PRESERVE_METADATA,
        quantity: Any = _PRESERVE_METADATA,
    ) -> Waveform:
        return Waveform(
            name=name,
            data=np.asarray(data),
            unit=self.unit if unit is _PRESERVE_METADATA else unit,
            quantity=self.quantity if quantity is _PRESERVE_METADATA else quantity,
            title=self.title,
            source_vector=self.raw_vector_name,
            abscissa=self.abscissa_name,
            abscissa_data=self.abscissa_data,
            metadata={**self.metadata, "derived_from": self.name},
            display_name=name,
            raw_vector_name=self.raw_vector_name,
            vector_kind=self.vector_kind,
            analysis=self.analysis,
            source=self.source,
            extraction=self.extraction,
            abscissa_name=self.abscissa_name,
            plot_name=self.plot_name,
        )


def _abscissa_unit(name: str) -> str | None:
    if name == "frequency":
        return "Hz"
    if name == "time":
        return "s"
    return None


def _abscissa_values(abscissa: Any) -> tuple[np.ndarray, Any]:
    if isinstance(abscissa, Waveform):
        return np.asarray(abscissa.data), abscissa.unit
    unit_array = _unit_array_abscissa_values(abscissa)
    if unit_array is not None:
        return unit_array
    return np.asarray(abscissa), None


def _waveform_abscissa_values(waveform: Waveform, abscissa: Any) -> tuple[np.ndarray, Any]:
    if abscissa is not None:
        return _abscissa_values(abscissa)
    if waveform.abscissa_data is None:
        raise ValueError(f"waveform {waveform.name} has no abscissa data")
    return np.asarray(waveform.abscissa_data), _abscissa_unit(str(waveform.abscissa_name or ""))


def _target_abscissa_values(abscissa: Any, fallback_name: str | None) -> tuple[np.ndarray, str]:
    if isinstance(abscissa, Waveform):
        return np.asarray(abscissa.data), abscissa.name
    unit_array = _unit_array_abscissa_values(abscissa)
    if unit_array is not None:
        values, _unit = unit_array
        return values, fallback_name or "abscissa"
    return np.asarray(abscissa), fallback_name or "abscissa"


def _sample_abscissa_values(abscissa: Any) -> tuple[np.ndarray, bool]:
    if isinstance(abscissa, Waveform):
        values = np.asarray(abscissa.data)
    else:
        unit_array = _unit_array_abscissa_values(abscissa)
        values = unit_array[0] if unit_array is not None else np.asarray(abscissa)
    if values.ndim == 0:
        return values.reshape(1), True
    return values, False


def _unit_array_abscissa_values(abscissa: Any) -> tuple[np.ndarray, str] | None:
    from monata.units import UnitArray

    if not isinstance(abscissa, UnitArray):
        return None
    return np.asarray(abscissa.values), abscissa.unit.symbol


def _waveform_summary(func: Any, waveform: Waveform, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    if kwargs.get("out") is not None:
        raise ValueError("Waveform summary operations do not support out")
    return _waveform_quantity_or_value(waveform, func(waveform.data, *args, **kwargs))


def _waveform_quantity_or_value(waveform: Waveform, value: Any) -> Any:
    if waveform.unit is None or np.iscomplexobj(value):
        return _plain_numpy_result(value)
    from monata.units import UnitError, quantity

    try:
        return quantity(value, str(waveform.unit))
    except (TypeError, ValueError, UnitError):
        return _plain_numpy_result(value)


def _plain_numpy_result(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _primary_waveform_input(inputs: tuple[Any, ...]) -> Waveform | None:
    return next((input_ for input_ in inputs if isinstance(input_, Waveform)), None)


def _waveform_ufunc_returns_array(ufunc: Any, result: Any) -> bool:
    if ufunc in _WAVEFORM_COMPARISON_UFUNCS:
        return True
    if isinstance(result, tuple):
        return True
    values = np.asarray(result)
    return values.dtype == np.dtype(bool)


def _waveform_unit_array_ufunc(ufunc: Any, inputs: tuple[Any, ...]) -> tuple[np.ndarray, str] | None:
    if not any(isinstance(input_, Waveform) and input_.unit is not None for input_ in inputs):
        return None
    from monata.units import Quantity, UnitArray, UnitError, quantity

    try:
        unit_inputs = []
        for input_ in inputs:
            if isinstance(input_, Waveform):
                if input_.unit is None:
                    return None
                unit_inputs.append(quantity(input_.data, str(input_.unit)))
            else:
                unit_inputs.append(input_)
        unit_result = ufunc(*unit_inputs)
    except (TypeError, ValueError, UnitError):
        return None
    if isinstance(unit_result, UnitArray):
        return np.asarray(unit_result.values), unit_result.unit.symbol
    if isinstance(unit_result, Quantity):
        return np.asarray(unit_result.value), unit_result.unit.symbol
    return None


def _waveform_ufunc_preserves_unit(ufunc: Any, inputs: tuple[Any, ...], primary: Waveform) -> bool:
    if primary.unit is None:
        return False
    waveforms = [input_ for input_ in inputs if isinstance(input_, Waveform)]
    if ufunc in _WAVEFORM_UNARY_PRESERVE_UFUNCS and len(waveforms) == 1:
        return True
    if ufunc not in _WAVEFORM_BINARY_PRESERVE_UFUNCS or len(inputs) != 2:
        return False
    if len(waveforms) == 1:
        return True
    if ufunc in {np.add, np.subtract}:
        return all(waveform.unit == primary.unit for waveform in waveforms)
    return False


def _waveform_ufunc_quantity(ufunc: Any, inputs: tuple[Any, ...], primary: Waveform) -> str | None:
    if primary.quantity is None:
        return None
    if ufunc in {np.multiply, np.divide, np.true_divide, np.square, np.sqrt, np.reciprocal}:
        return f"{primary.quantity}_{ufunc.__name__}"
    waveforms = [input_ for input_ in inputs if isinstance(input_, Waveform)]
    if len(waveforms) > 1 and any(waveform.quantity != primary.quantity for waveform in waveforms):
        return None
    return primary.quantity


def _waveform_ufunc_name(ufunc: Any, primary: Waveform) -> str:
    return f"{ufunc.__name__}({primary.name})"


def _coerce_abscissa_input(abscissa: Any, abscissa_name: str | None) -> tuple[str | None, np.ndarray | None]:
    if abscissa is None:
        return abscissa_name, None
    if isinstance(abscissa, Waveform):
        return abscissa_name or abscissa.name, np.asarray(abscissa.data)
    return abscissa_name or "abscissa", np.asarray(abscissa)


def _optional_abscissa_data(abscissa_data: Any, data: np.ndarray) -> np.ndarray | None:
    if abscissa_data is None:
        return None
    values = np.asarray(abscissa_data)
    if values.ndim != 1:
        raise ValueError("waveform abscissa_data must be one-dimensional")
    expected = data.shape[0] if data.ndim else data.size
    if values.shape[0] != expected:
        raise ValueError("waveform abscissa_data and data must have the same length")
    return values


def _sliced_abscissa_data(abscissa_data: np.ndarray | None, key: Any, data: np.ndarray) -> np.ndarray | None:
    if abscissa_data is None:
        return None
    values = np.asarray(abscissa_data)
    candidates = (key,)
    if isinstance(key, tuple) and key:
        candidates = (key, key[0])
    for candidate in candidates:
        try:
            sliced = np.asarray(values[candidate])
        except (IndexError, TypeError, ValueError):
            continue
        if sliced.ndim == 1 and data.ndim >= 1 and sliced.shape[0] == data.shape[0]:
            return sliced
    return None


def _validate_matching_abscissa(data: np.ndarray, abscissa: np.ndarray) -> None:
    if data.ndim != 1 or abscissa.ndim != 1:
        raise ValueError("waveform calculus requires one-dimensional data and abscissa")
    if data.shape[0] != abscissa.shape[0]:
        raise ValueError("waveform data and abscissa must have the same length")
    if data.shape[0] < 2:
        raise ValueError("waveform calculus requires at least two samples")


def _validate_interpolation_abscissas(
    data: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    operation: str,
) -> None:
    _validate_matching_abscissa(data, source)
    if target.ndim != 1:
        shape = "a one-dimensional" if operation == "resampling" else "a scalar or one-dimensional"
        raise ValueError(f"waveform {operation} requires {shape} target abscissa")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError(f"waveform {operation} requires finite abscissa values")
    if np.any(np.diff(source) <= 0):
        raise ValueError(f"waveform {operation} source abscissa must be strictly increasing")


def _validate_window_abscissa(data: np.ndarray, source: np.ndarray) -> None:
    if data.ndim != 1 or source.ndim != 1:
        raise ValueError("waveform windowing requires one-dimensional data and abscissa")
    if data.shape[0] != source.shape[0]:
        raise ValueError("waveform data and abscissa must have the same length")
    if data.shape[0] == 0:
        raise ValueError("waveform windowing requires at least one sample")
    if not np.all(np.isfinite(source)):
        raise ValueError("waveform windowing requires finite abscissa values")


def _window_bound_value(bound: Any, label: str, source_unit: Any) -> float:
    values = _real_window_values(_window_bound_values(bound, source_unit), f"{label} bound")
    if values.ndim == 0:
        scalar = values.item()
    elif values.ndim == 1 and values.size == 1:
        scalar = values[0]
    else:
        raise ValueError(f"waveform window {label} bound must be a scalar")
    value = float(scalar)
    if not np.isfinite(value):
        raise ValueError(f"waveform window {label} bound must be finite")
    return value


def _window_bound_values(bound: Any, source_unit: Any) -> np.ndarray:
    if isinstance(bound, Waveform):
        return np.asarray(bound.data)
    from monata.units import Quantity, UnitArray, UnitError

    try:
        if isinstance(bound, Quantity):
            converted = bound.to(source_unit) if source_unit is not None else bound
            return np.asarray(converted.value)
        if isinstance(bound, UnitArray):
            converted = bound.to(source_unit) if source_unit is not None else bound
            return np.asarray(converted.values)
    except UnitError as exc:
        raise ValueError("waveform window bound must use abscissa-compatible units") from exc
    return np.asarray(bound)


def _real_window_values(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        if not np.allclose(np.imag(array), 0.0):
            raise ValueError(f"waveform window {label} must use real values")
        array = np.real(array)
    try:
        return np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"waveform window {label} must use numeric values") from exc


def _source_abscissa_name(source_abscissa: Any, fallback_name: str | None) -> str:
    if isinstance(source_abscissa, Waveform):
        return source_abscissa.name
    return fallback_name or "abscissa"


def _interp_waveform_data(
    data: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    left: Any = None,
    right: Any = None,
) -> np.ndarray:
    if not np.iscomplexobj(data):
        return np.interp(target, source, data, left=left, right=right)
    left_value = None if left is None else complex(left)
    right_value = None if right is None else complex(right)
    real = np.interp(
        target,
        source,
        np.real(data),
        left=None if left_value is None else left_value.real,
        right=None if right_value is None else right_value.real,
    )
    imaginary = np.interp(
        target,
        source,
        np.imag(data),
        left=None if left_value is None else left_value.imag,
        right=None if right_value is None else right_value.imag,
    )
    return real + 1j * imaginary


def _derived_unit(unit: Any, abscissa_unit: Any, operator: str) -> str | None:
    if unit is None or abscissa_unit is None:
        return None
    return f"{unit}{operator}{abscissa_unit}"


def _derived_quantity(quantity: str | None, operation: str) -> str | None:
    if quantity is None:
        return None
    return f"{quantity}_{operation}"


__all__ = ["Waveform"]
