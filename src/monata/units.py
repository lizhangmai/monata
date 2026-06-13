"""Circuit-oriented unit values and SPICE suffix helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterator, Literal, Mapping, TypeAlias, overload

import numpy as np


class UnitError(ValueError):
    """Raised for invalid or incompatible unit operations."""


_SPICE_SUFFIXES: dict[str, float] = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "mil": 25.4e-6,
    "": 1.0,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}
_SPICE_RENDER_SUFFIXES: tuple[tuple[str, float], ...] = (
    ("t", 1e12),
    ("g", 1e9),
    ("meg", 1e6),
    ("k", 1e3),
    ("", 1.0),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
    ("a", 1e-18),
)
_SI_PREFIXES: dict[str, float] = {
    "Y": 1e24,
    "Z": 1e21,
    "E": 1e18,
    "P": 1e15,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "Meg": 1e6,
    "meg": 1e6,
    "k": 1e3,
    "K": 1e3,
    "h": 1e2,
    "da": 1e1,
    "": 1.0,
    "m": 1e-3,
    "u": 1e-6,
    "μ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
    "z": 1e-21,
    "y": 1e-24,
    "mil": 25.4e-6,
}
_SI_PREFIX_SYMBOLS: dict[str, str] = {
    "Meg": "Meg",
    "meg": "Meg",
    "K": "k",
    "μ": "u",
}
_SPICE_NUMBER_RE = re.compile(
    r"^\s*(?P<number>[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)(?P<suffix>[A-Za-z]*)\s*$"
)
_SPICE_RKM_NUMBER_RE = re.compile(
    r"^\s*(?P<whole>[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+)))(?P<marker>[A-Za-z])(?P<fraction>\d+)(?P<extra>[A-Za-z]*)\s*$"
)
SpiceNumberDialect: TypeAlias = Literal["standard", "ltspice"]
ScalarValue: TypeAlias = int | float | np.number
ArrayValue: TypeAlias = np.ndarray | list[Any] | tuple[Any, ...]


@dataclass(frozen=True)
class Unit:
    """A scale factor and dimension signature for circuit units."""

    name: str
    symbol: str
    factor: float = 1.0
    dimensions: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        dims = {name: power for name, power in dict(self.dimensions or {}).items() if power}
        object.__setattr__(self, "dimensions", dict(sorted(dims.items())))
        object.__setattr__(self, "factor", float(self.factor))

    def compatible_with(self, other: Unit) -> bool:
        return dict(self.dimensions or {}) == dict(other.dimensions or {})

    def conversion_factor_to(self, other: Unit) -> float:
        if not self.compatible_with(other):
            raise UnitError(f"incompatible units: {self.symbol} and {other.symbol}")
        return self.factor / other.factor

    @overload
    def quantity(self, value: ArrayValue) -> UnitArray: ...

    @overload
    def quantity(self, value: ScalarValue) -> Quantity: ...

    @overload
    def quantity(self, value: Any) -> Quantity | UnitArray: ...

    def quantity(self, value: Any) -> Quantity | UnitArray:
        if _is_array_like(value):
            return UnitArray(value, self)
        return Quantity(value, self)

    @overload
    def __call__(self, value: ArrayValue) -> UnitArray: ...

    @overload
    def __call__(self, value: ScalarValue) -> Quantity: ...

    @overload
    def __call__(self, value: Any) -> Quantity | UnitArray: ...

    def __call__(self, value: Any) -> Quantity | UnitArray:
        return self.quantity(value)

    @overload
    def __rmatmul__(self, value: ArrayValue) -> UnitArray: ...

    @overload
    def __rmatmul__(self, value: ScalarValue) -> Quantity: ...

    @overload
    def __rmatmul__(self, value: Any) -> Quantity | UnitArray: ...

    def __rmatmul__(self, value: Any) -> Quantity | UnitArray:
        return self.quantity(value)

    def __mul__(self, other: Unit) -> Unit:
        return _derived_unit(self, other, "*")

    def __truediv__(self, other: Unit) -> Unit:
        return _derived_unit(self, other, "/")

    def reciprocal(self) -> Unit:
        return _DIMENSIONLESS / self

    def __pow__(self, power: int) -> Unit:
        exponent = _unit_power_exponent(power)
        if exponent == 0:
            return _DIMENSIONLESS
        dimensions = {name: power * exponent for name, power in dict(self.dimensions or {}).items()}
        return _known_or_derived_unit(f"{self.symbol}^{exponent}", self.factor**exponent, dimensions)

    def square(self) -> Unit:
        return self**2

    def sqrt(self) -> Unit:
        dimensions: dict[str, int] = {}
        for name, power in dict(self.dimensions or {}).items():
            if power % 2:
                raise UnitError(f"{self.symbol} cannot be square-rooted with integer unit dimensions")
            dimensions[name] = power // 2
        if not dimensions and np.isclose(math.sqrt(self.factor), 1.0):
            return _DIMENSIONLESS
        return _known_or_derived_unit(f"sqrt({self.symbol})", math.sqrt(self.factor), dimensions)


@dataclass(frozen=True)
class Quantity:
    """A scalar numeric value carrying a unit."""

    value: float
    unit: Unit

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))

    def to(self, target: Unit | str) -> Quantity:
        target_unit = unit(target)
        return Quantity(self.value * self.unit.conversion_factor_to(target_unit), target_unit)

    def clone(self) -> Quantity:
        """Return an independent quantity with the same value and unit."""

        return Quantity(self.value, self.unit)

    def clone_prefixed_unit(self, value: int | float | np.number) -> Quantity:
        """Return a quantity with this unit and a replacement numeric value."""

        return Quantity(float(value), self.unit)

    @property
    def prefixed_unit(self) -> Unit:
        """Return the concrete scaled unit carried by this quantity."""

        return self.unit

    @property
    def scale(self) -> float:
        """Return the scale factor from this unit to its base dimensions."""

        return self.unit.factor

    def get_prefixed_unit(self, power: int = 0) -> Unit:
        """Return a registered compatible unit with the requested SI prefix power."""

        return _prefixed_unit_for_power(self.unit, power)

    def convert_to_power(self, power: int = 0) -> Quantity:
        """Convert to a compatible unit whose scale factor is 10**power."""

        return self.to(self.get_prefixed_unit(power))

    def canonise(self) -> Quantity:
        """Return this quantity using an engineering prefix when one is registered."""

        base_value = abs(self.value * self.unit.factor)
        if base_value == 0:
            return self
        target_power = _engineering_power(base_value)
        try:
            return self.convert_to_power(target_power)
        except UnitError:
            return self

    def compatible_with(self, other: Quantity | Unit) -> bool:
        other_unit = other.unit if isinstance(other, Quantity) else other
        return self.unit.compatible_with(other_unit)

    def is_same_unit(self, other: Quantity | UnitArray | Unit | str) -> bool:
        """Return whether another value shares compatible physical dimensions."""

        return self.unit.compatible_with(_unit_from_metadata_operand(other))

    def is_same_power(self, other: Quantity | UnitArray | Unit | str) -> bool:
        """Return whether another value uses the same concrete scaled unit."""

        return self.unit == _unit_from_metadata_operand(other)

    def spice(self) -> str:
        return format_spice_number(self.value * self.unit.factor)

    def str(self, spice: bool = False, space: bool = False, unit: bool = True) -> str:
        if spice:
            return self.spice()
        text = _format_float(self.value)
        if not unit or not self.unit.symbol:
            return text
        separator = " " if space else ""
        return f"{text}{separator}{self.unit.symbol}"

    def str_space(self) -> str:
        return self.str(space=True)

    @property
    def frequency(self) -> Quantity:
        if self.unit.compatible_with(s):
            return Quantity(1 / self.to(s).value, Hz)
        if self.unit.compatible_with(Hz):
            return self.to(Hz)
        raise UnitError(f"{self.unit.symbol} cannot be converted to frequency")

    @property
    def period(self) -> Quantity:
        if self.unit.compatible_with(Hz):
            return Quantity(1 / self.to(Hz).value, s)
        if self.unit.compatible_with(s):
            return self.to(s)
        raise UnitError(f"{self.unit.symbol} cannot be converted to period")

    @property
    def pulsation(self) -> Quantity:
        frequency = self.frequency
        return Quantity(2 * math.pi * frequency.value, Unit("radian per second", "rad/s", 1.0, {"time": -1}))

    def reciprocal(self) -> Quantity:
        return Quantity(1 / self.value, self.unit.reciprocal())

    def square(self) -> Quantity:
        return self**2

    def sqrt(self) -> Quantity:
        return Quantity(math.sqrt(self.value), self.unit.sqrt())

    def __bool__(self) -> bool:
        return self.value != 0

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Quantity):
            if not self.compatible_with(other):
                return False
            return self.value == other.to(self.unit).value
        if isinstance(other, (int, float, np.number)) and not isinstance(other, bool):
            return self.value == float(other)
        return NotImplemented

    def __lt__(self, other: Quantity | int | float) -> bool:
        return self.value < _quantity_value_in_unit(other, self.unit)

    def __le__(self, other: Quantity | int | float) -> bool:
        return self.value <= _quantity_value_in_unit(other, self.unit)

    def __gt__(self, other: Quantity | int | float) -> bool:
        return self.value > _quantity_value_in_unit(other, self.unit)

    def __ge__(self, other: Quantity | int | float) -> bool:
        return self.value >= _quantity_value_in_unit(other, self.unit)

    def __neg__(self) -> Quantity:
        return Quantity(-self.value, self.unit)

    def __pos__(self) -> Quantity:
        return Quantity(+self.value, self.unit)

    def __abs__(self) -> Quantity:
        return Quantity(abs(self.value), self.unit)

    def __pow__(self, power: int) -> Quantity:
        exponent = _unit_power_exponent(power)
        return Quantity(self.value**exponent, self.unit**exponent)

    def __add__(self, other: Quantity | int | float) -> Quantity:
        return _quantity_add(self, other, 1)

    def __radd__(self, other: Quantity | int | float) -> Quantity:
        return self.__add__(other)

    def __sub__(self, other: Quantity | int | float) -> Quantity:
        return _quantity_add(self, other, -1)

    def __rsub__(self, other: Quantity | int | float) -> Quantity:
        return Quantity(float(other), self.unit) - self if _is_scalar(other) else NotImplemented

    def __mul__(self, other: Quantity | int | float) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.unit * other.unit)
        if _is_scalar(other):
            return Quantity(self.value * float(other), self.unit)
        return NotImplemented

    def __rmul__(self, other: Quantity | int | float) -> Quantity:
        return self.__mul__(other)

    def __truediv__(self, other: Quantity | int | float) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.unit / other.unit)
        if _is_scalar(other):
            return Quantity(self.value / float(other), self.unit)
        return NotImplemented

    def __rtruediv__(self, other: int | float) -> Quantity:
        if _is_scalar(other):
            return Quantity(float(other) / self.value, _DIMENSIONLESS / self.unit)
        return NotImplemented


@dataclass(frozen=True)
class UnitArray:
    """A numpy-backed array carrying one unit for all values."""

    __array_priority__ = 1000

    values: np.ndarray
    unit: Unit

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.asarray(self.values))

    def to(self, target: Unit | str) -> UnitArray:
        target_unit = unit(target)
        return UnitArray(self.values * self.unit.conversion_factor_to(target_unit), target_unit)

    @property
    def prefixed_unit(self) -> Unit:
        """Return the concrete scaled unit carried by this array."""

        return self.unit

    @property
    def scale(self) -> float:
        """Return the scale factor from this unit to its base dimensions."""

        return self.unit.factor

    def get_prefixed_unit(self, power: int = 0) -> Unit:
        """Return a registered compatible unit with the requested SI prefix power."""

        return _prefixed_unit_for_power(self.unit, power)

    def convert_to_power(self, power: int = 0) -> UnitArray:
        """Convert to a compatible unit whose scale factor is 10**power."""

        return self.to(self.get_prefixed_unit(power))

    def compatible_with(self, other: UnitArray | Unit) -> bool:
        other_unit = other.unit if isinstance(other, UnitArray) else other
        return self.unit.compatible_with(other_unit)

    def is_same_unit(self, other: Quantity | UnitArray | Unit | str) -> bool:
        """Return whether another value shares compatible physical dimensions."""

        return self.unit.compatible_with(_unit_from_metadata_operand(other))

    def is_same_power(self, other: Quantity | UnitArray | Unit | str) -> bool:
        """Return whether another value uses the same concrete scaled unit."""

        return self.unit == _unit_from_metadata_operand(other)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def as_array(self, *, copy: bool = False) -> np.ndarray:
        return np.array(self.values, copy=copy)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[Quantity | UnitArray]:
        for value in self.values:
            yield _unit_array_item(value, self.unit)

    def __getitem__(self, index: Any) -> Quantity | UnitArray:
        return _unit_array_item(self.values[index], self.unit)

    def __setitem__(self, index: Any, value: Any) -> None:
        self.values[index] = _unit_array_assignment_value(value, self.unit)

    def sum(self, *args: Any, **kwargs: Any) -> Quantity | UnitArray:
        return _unit_array_summary(np.sum, self, args, kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> Quantity | UnitArray:
        return _unit_array_summary(np.mean, self, args, kwargs)

    def min(self, *args: Any, **kwargs: Any) -> Quantity | UnitArray:
        return _unit_array_summary(np.min, self, args, kwargs)

    def max(self, *args: Any, **kwargs: Any) -> Quantity | UnitArray:
        return _unit_array_summary(np.max, self, args, kwargs)

    def reciprocal(self) -> UnitArray:
        return UnitArray(1 / self.values, self.unit.reciprocal())

    def square(self) -> UnitArray:
        return self**2

    def sqrt(self) -> UnitArray:
        return UnitArray(np.sqrt(self.values), self.unit.sqrt())

    @property
    def frequency(self) -> UnitArray:
        if self.unit.compatible_with(s):
            return UnitArray(1 / self.to(s).values, Hz)
        if self.unit.compatible_with(Hz):
            return self.to(Hz)
        raise UnitError(f"{self.unit.symbol} cannot be converted to frequency")

    @property
    def period(self) -> UnitArray:
        if self.unit.compatible_with(Hz):
            return UnitArray(1 / self.to(Hz).values, s)
        if self.unit.compatible_with(s):
            return self.to(s)
        raise UnitError(f"{self.unit.symbol} cannot be converted to period")

    @property
    def pulsation(self) -> UnitArray:
        frequency = self.frequency
        return UnitArray(2 * math.pi * frequency.values, Unit("radian per second", "rad/s", 1.0, {"time": -1}))

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if kwargs.get("out") is not None:
            return NotImplemented
        if method == "reduce" and ufunc in _UNIT_ARRAY_REDUCE_PRESERVE and len(inputs) == 1:
            value = _require_unit_array_input(inputs[0])
            return _unit_array_item(ufunc.reduce(value.values, **kwargs), value.unit)
        if method != "__call__":
            return NotImplemented
        if ufunc == np.reciprocal and len(inputs) == 1:
            value = _require_unit_array_input(inputs[0])
            return value.reciprocal()
        if ufunc == np.square and len(inputs) == 1:
            value = _require_unit_array_input(inputs[0])
            return value.square()
        if ufunc == np.sqrt and len(inputs) == 1:
            value = _require_unit_array_input(inputs[0])
            return value.sqrt()
        if ufunc == np.power and len(inputs) == 2 and isinstance(inputs[0], UnitArray):
            return inputs[0] ** inputs[1]
        if ufunc in _UNIT_ARRAY_UNARY_PRESERVE and len(inputs) == 1:
            value = _require_unit_array_input(inputs[0])
            return UnitArray(ufunc(value.values, **kwargs), value.unit)
        if ufunc in _UNIT_ARRAY_ADD_SUBTRACT and len(inputs) == 2:
            return _unit_array_ufunc_add_subtract(ufunc, inputs[0], inputs[1], kwargs)
        if ufunc in _UNIT_ARRAY_MULTIPLY_DIVIDE and len(inputs) == 2:
            return _unit_array_ufunc_multiply_divide(ufunc, inputs[0], inputs[1])
        if ufunc in _UNIT_ARRAY_COMPARISON and len(inputs) == 2:
            return _unit_array_ufunc_compare(ufunc, inputs[0], inputs[1], kwargs)
        if ufunc in _UNIT_ARRAY_EXTREMA and len(inputs) == 2:
            return _unit_array_ufunc_extrema(ufunc, inputs[0], inputs[1], kwargs)
        return NotImplemented

    def __add__(self, other: UnitArray | Quantity | int | float) -> UnitArray:
        return _unit_array_add(self, other, 1)

    def __radd__(self, other: UnitArray | Quantity | int | float) -> UnitArray:
        return self.__add__(other)

    def __sub__(self, other: UnitArray | Quantity | int | float) -> UnitArray:
        return _unit_array_add(self, other, -1)

    def __mul__(self, other: UnitArray | Quantity | int | float) -> UnitArray:
        if isinstance(other, UnitArray):
            return UnitArray(self.values * other.values, self.unit * other.unit)
        if isinstance(other, Quantity):
            return UnitArray(self.values * other.value, self.unit * other.unit)
        if _is_scalar(other):
            return UnitArray(self.values * float(other), self.unit)
        return NotImplemented

    def __rmul__(self, other: UnitArray | Quantity | int | float) -> UnitArray:
        return self.__mul__(other)

    def __truediv__(self, other: UnitArray | Quantity | int | float) -> UnitArray:
        if isinstance(other, UnitArray):
            return UnitArray(self.values / other.values, self.unit / other.unit)
        if isinstance(other, Quantity):
            return UnitArray(self.values / other.value, self.unit / other.unit)
        if _is_scalar(other):
            return UnitArray(self.values / float(other), self.unit)
        return NotImplemented

    def __pow__(self, power: int) -> UnitArray:
        exponent = _unit_power_exponent(power)
        return UnitArray(self.values**exponent, self.unit**exponent)


def parse_spice_number(value: str | int | float, *, dialect: SpiceNumberDialect = "standard") -> float:
    """Parse a numeric value with an optional SPICE suffix.

    `dialect="ltspice"` additionally accepts RKM forms such as `2k3`.
    """

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    _validate_spice_number_dialect(dialect)
    text = str(value).strip()
    if not text:
        raise UnitError("empty SPICE number")
    if dialect == "ltspice":
        rkm_value = _parse_rkm_number(text)
        if rkm_value is not None:
            return rkm_value
    match = _SPICE_NUMBER_RE.match(text)
    if match is None:
        raise UnitError(f"invalid SPICE number: {value}")
    numeric = match.group("number")
    suffix = _spice_suffix(match.group("suffix"))
    try:
        base = float(numeric)
    except ValueError as exc:
        raise UnitError(f"invalid SPICE number: {value}") from exc
    return base * _SPICE_SUFFIXES[suffix]


def _validate_spice_number_dialect(dialect: str) -> None:
    if dialect not in {"standard", "ltspice"}:
        raise UnitError(f"unknown SPICE number dialect: {dialect}")


def _parse_rkm_number(text: str) -> float | None:
    match = _SPICE_RKM_NUMBER_RE.match(text)
    if match is None:
        return None
    marker = match.group("marker")
    suffix = "" if marker.lower() == "r" else _spice_suffix(marker)
    if marker.lower() != "r" and not suffix:
        return None
    whole = match.group("whole")
    sign = "-" if whole.startswith("-") else ""
    unsigned_whole = whole.lstrip("+-")
    number = float(f"{sign}{unsigned_whole}.{match.group('fraction')}")
    return number * _SPICE_SUFFIXES[suffix]


def _spice_suffix(text: str) -> str:
    lower = text.lower()
    if lower.startswith("meg"):
        return "meg"
    if lower.startswith("mil"):
        return "mil"
    if text.startswith("a"):
        return "a"
    if lower[:1] in {"t", "g", "k", "m", "u", "n", "p", "f"}:
        return lower[:1]
    return ""


def format_spice_number(value: int | float) -> str:
    """Render a numeric value with a compact SPICE suffix when practical."""

    number = float(value)
    if number == 0:
        return "0"
    for suffix, factor in _SPICE_RENDER_SUFFIXES:
        scaled = number / factor
        if 1 <= abs(scaled) < 1000:
            return f"{_format_float(scaled)}{suffix}"
    return _format_float(number)


def format_spice_value(value: Any, *, none: str = "") -> str:
    """Render a scalar Python or Monata unit value as one SPICE token."""

    if value is None:
        return none
    if isinstance(value, Quantity):
        return value.spice()
    if isinstance(value, UnitArray):
        raise UnitError("SPICE value rendering expects a scalar value, got UnitArray")
    return str(value)


def format_spice_values(*values: Any, none: str = "") -> tuple[str, ...]:
    """Render multiple scalar values with `format_spice_value`."""

    return tuple(format_spice_value(value, none=none) for value in values)


def join_spice_values(*values: Any, none: str = "", sep: str = " ") -> str:
    """Render scalar values as SPICE tokens and join non-empty tokens."""

    return sep.join(token for token in format_spice_values(*values, none=none) if token)


def unit(value: Unit | str) -> Unit:
    """Return a known unit by symbol or pass through a Unit instance."""

    if isinstance(value, Unit):
        return value
    try:
        return _UNITS[str(value)]
    except KeyError as exc:
        raise UnitError(f"unknown unit: {value}") from exc


@overload
def quantity(value: ArrayValue, value_unit: Unit | str) -> UnitArray: ...


@overload
def quantity(value: ScalarValue, value_unit: Unit | str) -> Quantity: ...


@overload
def quantity(value: Any, value_unit: Unit | str) -> Quantity | UnitArray: ...


def quantity(value: Any, value_unit: Unit | str) -> Quantity | UnitArray:
    return unit(value_unit).quantity(value)


@overload
def unit_value(value: ArrayValue) -> UnitArray: ...


@overload
def unit_value(value: ScalarValue) -> Quantity: ...


@overload
def unit_value(value: Any) -> Quantity | UnitArray: ...


def unit_value(value: Any) -> Quantity | UnitArray:
    return _DIMENSIONLESS.quantity(value)


def as_unit(value: Any, value_unit: Unit | str, *, none: bool = False) -> Quantity | UnitArray | None:
    if none and value is None:
        return None
    target = unit(value_unit)
    if isinstance(value, (Quantity, UnitArray)):
        return value.to(target)
    return target.quantity(value)


def scaled_unit(base: Unit | str, prefix: str, *, symbol: str | None = None, name: str | None = None) -> Unit:
    """Return a unit scaled by a known engineering/SI prefix."""

    base_unit = unit(base)
    prefix_symbol = _SI_PREFIX_SYMBOLS.get(prefix, prefix)
    unit_symbol = symbol if symbol is not None else f"{prefix_symbol}{base_unit.symbol}"
    unit_name = name if name is not None else f"{prefix or 'unit'}{base_unit.name}"
    return Unit(unit_name, unit_symbol, base_unit.factor * si_prefix(prefix), base_unit.dimensions)


def rms_to_amplitude(value: Any) -> Any:
    """Convert a sinusoid RMS level to peak amplitude."""

    return _scale_level(value, math.sqrt(2))


def amplitude_to_rms(value: Any) -> Any:
    """Convert a sinusoid peak amplitude to RMS level."""

    return _scale_level(value, 1 / math.sqrt(2))


def si_prefix(name: str) -> float:
    try:
        return _SI_PREFIXES[name]
    except KeyError as exc:
        raise UnitError(f"unknown SI prefix: {name}") from exc


def yotta(value: Any) -> Any:
    return _prefix_value(value, "Y")


def zetta(value: Any) -> Any:
    return _prefix_value(value, "Z")


def exa(value: Any) -> Any:
    return _prefix_value(value, "E")


def peta(value: Any) -> Any:
    return _prefix_value(value, "P")


def tera(value: Any) -> Any:
    return _prefix_value(value, "T")


def giga(value: Any) -> Any:
    return _prefix_value(value, "G")


def mega(value: Any) -> Any:
    return _prefix_value(value, "M")


def kilo(value: Any) -> Any:
    return _prefix_value(value, "k")


def hecto(value: Any) -> Any:
    return _prefix_value(value, "h")


def deca(value: Any) -> Any:
    return _prefix_value(value, "da")


def milli(value: Any) -> Any:
    return _prefix_value(value, "m")


def micro(value: Any) -> Any:
    return _prefix_value(value, "u")


def nano(value: Any) -> Any:
    return _prefix_value(value, "n")


def pico(value: Any) -> Any:
    return _prefix_value(value, "p")


def femto(value: Any) -> Any:
    return _prefix_value(value, "f")


def atto(value: Any) -> Any:
    return _prefix_value(value, "a")


def zepto(value: Any) -> Any:
    return _prefix_value(value, "z")


def yocto(value: Any) -> Any:
    return _prefix_value(value, "y")


def _prefix_value(value: Any, prefix: str) -> Any:
    return _scale_level(value, si_prefix(prefix))


def _format_float(value: float) -> str:
    text = f"{value:.12g}"
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        exponent = exponent.lstrip("+").lstrip("0") or "0"
        return f"{mantissa}e{exponent}"
    return text


def _is_array_like(value: Any) -> bool:
    return isinstance(value, np.ndarray) or (
        isinstance(value, (list, tuple)) and not all(isinstance(item, (str, bytes)) for item in value)
    )


rad = Unit("radian", "rad", 1.0, {})
deg = Unit("degree", "deg", math.pi / 180, {})
sr = Unit("steradian", "sr", 1.0, {})
m = Unit("metre", "m", 1.0, {"length": 1})
mm = scaled_unit(m, "m", name="millimetre")
um = scaled_unit(m, "u", name="micrometre")
kg = Unit("kilogram", "kg", 1.0, {"mass": 1})
K = Unit("kelvin", "K", 1.0, {"temperature": 1})
Degree = Unit("degree celsius", "Degree", 1.0, {"temperature": 1})
mol = Unit("mole", "mol", 1.0, {"substance": 1})
cd = Unit("candela", "cd", 1.0, {"luminous_intensity": 1})
V = Unit("volt", "V", 1.0, {"voltage": 1})
mV = scaled_unit(V, "m", name="millivolt")
A = Unit("ampere", "A", 1.0, {"current": 1})
mA = scaled_unit(A, "m", name="milliampere")
C = Unit("coulomb", "C", 1.0, {"current": 1, "time": 1})
uC = scaled_unit(C, "u", name="microcoulomb")
nC = scaled_unit(C, "n", name="nanocoulomb")
Ohm = Unit("ohm", "Ohm", 1.0, {"current": -1, "voltage": 1})
kOhm = scaled_unit(Ohm, "k", name="kilohm")
S = Unit("siemens", "S", 1.0, {"current": 1, "voltage": -1})
mS = scaled_unit(S, "m", name="millisiemens")
uS = scaled_unit(S, "u", name="microsiemens")
F = Unit("farad", "F", 1.0, {"current": 1, "time": 1, "voltage": -1})
uF = scaled_unit(F, "u", name="microfarad")
nF = scaled_unit(F, "n", name="nanofarad")
pF = scaled_unit(F, "p", name="picofarad")
H = Unit("henry", "H", 1.0, {"current": -1, "time": 1, "voltage": 1})
uH = scaled_unit(H, "u", name="microhenry")
nH = scaled_unit(H, "n", name="nanohenry")
s = Unit("second", "s", 1.0, {"time": 1})
ms = scaled_unit(s, "m", name="millisecond")
us = scaled_unit(s, "u", name="microsecond")
ns = scaled_unit(s, "n", name="nanosecond")
Hz = Unit("hertz", "Hz", 1.0, {"time": -1})
kHz = scaled_unit(Hz, "k", name="kilohertz")
MHz = scaled_unit(Hz, "M", name="megahertz")
GHz = scaled_unit(Hz, "G", name="gigahertz")
W = Unit("watt", "W", 1.0, {"current": 1, "voltage": 1})
J = Unit("joule", "J", 1.0, {"current": 1, "time": 1, "voltage": 1})
N = Unit("newton", "N", 1.0, {"current": 1, "length": -1, "time": 1, "voltage": 1})
Pa = Unit("pascal", "Pa", 1.0, {"current": 1, "length": -3, "time": 1, "voltage": 1})
Wb = Unit("weber", "Wb", 1.0, {"time": 1, "voltage": 1})
T = Unit("tesla", "T", 1.0, {"length": -2, "time": 1, "voltage": 1})
Bq = Unit("becquerel", "Bq", 1.0, {"time": -1})
lm = Unit("lumen", "lm", 1.0, {"luminous_intensity": 1})
lx = Unit("lux", "lx", 1.0, {"length": -2, "luminous_intensity": 1})
Gy = Unit("gray", "Gy", 1.0, {"current": 1, "mass": -1, "time": 1, "voltage": 1})
Sv = Unit("sievert", "Sv", 1.0, {"current": 1, "mass": -1, "time": 1, "voltage": 1})
kat = Unit("katal", "kat", 1.0, {"substance": 1, "time": -1})
_DIMENSIONLESS = Unit("dimensionless", "", 1.0, {})

_KNOWN_UNITS = (
    rad,
    deg,
    sr,
    m,
    mm,
    um,
    kg,
    K,
    Degree,
    mol,
    cd,
    V,
    mV,
    A,
    mA,
    C,
    uC,
    nC,
    Ohm,
    kOhm,
    S,
    mS,
    uS,
    F,
    uF,
    nF,
    pF,
    H,
    uH,
    nH,
    s,
    ms,
    us,
    ns,
    Hz,
    kHz,
    MHz,
    GHz,
    W,
    J,
    N,
    Pa,
    Wb,
    T,
    Bq,
    lm,
    lx,
    Gy,
    Sv,
    kat,
)
_UNITS = {item.symbol: item for item in _KNOWN_UNITS}
_UNITS.update({item.name: item for item in _KNOWN_UNITS})
_UNITS.update({"meter": m, "micrometer": um, "ohm": Ohm, "Ω": Ohm, "°C": Degree, "degC": Degree})

_UNIT_ARRAY_UNARY_PRESERVE = {
    np.negative,
    np.positive,
    np.absolute,
    np.fabs,
    np.rint,
    np.floor,
    np.ceil,
    np.trunc,
}
_UNIT_ARRAY_ADD_SUBTRACT = {np.add, np.subtract}
_UNIT_ARRAY_MULTIPLY_DIVIDE = {np.multiply, np.divide, np.true_divide}
_UNIT_ARRAY_COMPARISON = {np.greater, np.greater_equal, np.less, np.less_equal, np.equal, np.not_equal}
_UNIT_ARRAY_EXTREMA = {np.maximum, np.minimum, np.fmax, np.fmin}
_UNIT_ARRAY_REDUCE_PRESERVE = {np.add, *_UNIT_ARRAY_EXTREMA}


def _quantity_add(left: Quantity, right: Quantity | int | float, sign: int) -> Quantity:
    if isinstance(right, Quantity):
        return Quantity(left.value + sign * right.to(left.unit).value, left.unit)
    if _is_scalar(right):
        return Quantity(left.value + sign * float(right), left.unit)
    return NotImplemented


def _quantity_value_in_unit(value: Quantity | int | float, value_unit: Unit) -> float:
    if isinstance(value, Quantity):
        return value.to(value_unit).value
    if _is_scalar(value):
        return float(value)
    raise TypeError("quantity comparison expects a Quantity or scalar")


def _unit_power_exponent(power: Any) -> int:
    if isinstance(power, (int, np.integer)) and not isinstance(power, bool):
        return int(power)
    raise UnitError("unit powers must be integers")


def _unit_array_add(left: UnitArray, right: UnitArray | Quantity | int | float, sign: int) -> UnitArray:
    if isinstance(right, UnitArray):
        return UnitArray(left.values + sign * right.to(left.unit).values, left.unit)
    if isinstance(right, Quantity):
        return UnitArray(left.values + sign * right.to(left.unit).value, left.unit)
    if _is_scalar(right):
        return UnitArray(left.values + sign * float(right), left.unit)
    return NotImplemented


def _unit_array_item(value: Any, value_unit: Unit) -> Quantity | UnitArray:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return Quantity(value.item(), value_unit)
        return UnitArray(value, value_unit)
    if isinstance(value, np.generic):
        return Quantity(value.item(), value_unit)
    if _is_scalar(value):
        return Quantity(value, value_unit)
    return UnitArray(value, value_unit)


def _unit_from_metadata_operand(value: Quantity | UnitArray | Unit | str) -> Unit:
    if isinstance(value, (Quantity, UnitArray)):
        return value.unit
    return unit(value)


def _prefixed_unit_for_power(source_unit: Unit, power: int) -> Unit:
    try:
        power_int = int(power)
    except (TypeError, ValueError) as exc:
        raise UnitError(f"invalid SI prefix power: {power}") from exc
    if power_int != power:
        raise UnitError(f"invalid SI prefix power: {power}")
    target_factor = 10.0**power_int
    probe = Unit("", "", target_factor, source_unit.dimensions)
    for known in _KNOWN_UNITS:
        if known.compatible_with(probe) and np.isclose(known.factor, target_factor):
            return known
    raise UnitError(f"no registered unit for {source_unit.symbol} at SI prefix power {power_int}")


def _engineering_power(base_value: float) -> int:
    log = math.log(base_value) / math.log(1000)
    power = int(log)
    if base_value < 1 and log != int(log):
        power -= 1
    return 3 * power


def _unit_array_assignment_value(value: Any, target_unit: Unit) -> Any:
    if isinstance(value, UnitArray):
        return value.to(target_unit).values
    if isinstance(value, Quantity):
        return value.to(target_unit).value
    return value


def _unit_array_summary(func: Any, value: UnitArray, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Quantity | UnitArray:
    if kwargs.get("out") is not None:
        raise UnitError("UnitArray summary operations do not support out")
    return _unit_array_item(func(value.values, *args, **kwargs), value.unit)


def _require_unit_array_input(value: Any) -> UnitArray:
    if isinstance(value, UnitArray):
        return value
    raise UnitError("numpy unit operation expects a UnitArray input")


def _unit_array_ufunc_add_subtract(ufunc: Any, left: Any, right: Any, kwargs: Mapping[str, Any]) -> Any:
    if ufunc == np.add:
        if isinstance(left, UnitArray):
            return _unit_array_add(left, right, 1)
        if isinstance(right, UnitArray):
            return _unit_array_add(right, left, 1)
    if ufunc == np.subtract:
        if isinstance(left, UnitArray):
            return _unit_array_add(left, right, -1)
        if isinstance(right, UnitArray):
            values = _value_in_unit(left, right.unit) - right.values
            return UnitArray(np.subtract(values, 0, **kwargs), right.unit)
    return NotImplemented


def _unit_array_ufunc_multiply_divide(ufunc: Any, left: Any, right: Any) -> Any:
    if ufunc == np.multiply:
        if isinstance(left, UnitArray):
            return left * right
        if isinstance(right, UnitArray):
            return right * left
    if ufunc in {np.divide, np.true_divide}:
        if isinstance(left, UnitArray):
            return left / right
        if isinstance(right, UnitArray):
            if isinstance(left, Quantity):
                return UnitArray(left.value / right.values, left.unit / right.unit)
            if _is_scalar(left):
                return UnitArray(float(left) / right.values, _DIMENSIONLESS / right.unit)
    return NotImplemented


def _unit_array_ufunc_compare(ufunc: Any, left: Any, right: Any, kwargs: Mapping[str, Any]) -> Any:
    target_unit = _unit_array_target_unit(left, right)
    left_values = _value_in_unit(left, target_unit)
    right_values = _value_in_unit(right, target_unit)
    return ufunc(left_values, right_values, **kwargs)


def _unit_array_ufunc_extrema(ufunc: Any, left: Any, right: Any, kwargs: Mapping[str, Any]) -> Any:
    target_unit = _unit_array_target_unit(left, right)
    left_values = _value_in_unit(left, target_unit)
    right_values = _value_in_unit(right, target_unit)
    return UnitArray(ufunc(left_values, right_values, **kwargs), target_unit)


def _unit_array_target_unit(left: Any, right: Any) -> Unit:
    if isinstance(left, UnitArray):
        return left.unit
    if isinstance(right, UnitArray):
        return right.unit
    raise UnitError("numpy unit operation expects at least one UnitArray input")


def _value_in_unit(value: Any, target_unit: Unit) -> Any:
    if isinstance(value, UnitArray):
        return value.to(target_unit).values
    if isinstance(value, Quantity):
        return value.to(target_unit).value
    if _is_scalar(value):
        return float(value)
    return NotImplemented


def _derived_unit(left: Unit, right: Unit, operator: str) -> Unit:
    if operator == "*":
        factor = left.factor * right.factor
        dimensions = _combine_dimensions(left.dimensions, right.dimensions, 1)
        symbol = _join_unit_symbol(left.symbol, right.symbol, "*")
    elif operator == "/":
        factor = left.factor / right.factor
        dimensions = _combine_dimensions(left.dimensions, right.dimensions, -1)
        symbol = _join_unit_symbol(left.symbol, right.symbol, "/")
    else:
        raise UnitError(f"unsupported unit operator: {operator}")
    return _known_or_derived_unit(symbol, factor, dimensions)


def _combine_dimensions(
    left: Mapping[str, int] | None,
    right: Mapping[str, int] | None,
    right_sign: int,
) -> dict[str, int]:
    result = dict(left or {})
    for name, exponent in dict(right or {}).items():
        result[name] = result.get(name, 0) + right_sign * exponent
        if result[name] == 0:
            del result[name]
    return dict(sorted(result.items()))


def _known_or_derived_unit(symbol: str, factor: float, dimensions: Mapping[str, int] | None) -> Unit:
    normalized = {name: power for name, power in dict(dimensions or {}).items() if power}
    if not normalized and np.isclose(factor, 1.0):
        return _DIMENSIONLESS
    for known in _UNITS.values():
        if known.compatible_with(Unit("", "", factor, normalized)) and np.isclose(known.factor, factor):
            return known
    return Unit(symbol or "dimensionless", symbol, factor, normalized)


def _join_unit_symbol(left: str, right: str, operator: str) -> str:
    if not left:
        return right if operator == "*" else f"1/{right}"
    if not right:
        return left
    return f"{left}{operator}{right}"


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _scale_level(value: Any, factor: float) -> Any:
    if isinstance(value, (Quantity, UnitArray)):
        return value * factor
    if _is_scalar(value):
        return float(value) * factor
    return np.asarray(value) * factor


__all__ = [
    "UnitError",
    "Unit",
    "Quantity",
    "UnitArray",
    "parse_spice_number",
    "format_spice_number",
    "format_spice_value",
    "format_spice_values",
    "join_spice_values",
    "si_prefix",
    "yotta",
    "zetta",
    "exa",
    "peta",
    "tera",
    "giga",
    "mega",
    "kilo",
    "hecto",
    "deca",
    "milli",
    "micro",
    "nano",
    "pico",
    "femto",
    "atto",
    "zepto",
    "yocto",
    "scaled_unit",
    "unit",
    "quantity",
    "unit_value",
    "as_unit",
    "rms_to_amplitude",
    "amplitude_to_rms",
    "rad",
    "deg",
    "sr",
    "m",
    "mm",
    "um",
    "kg",
    "K",
    "Degree",
    "mol",
    "cd",
    "V",
    "mV",
    "A",
    "mA",
    "C",
    "uC",
    "nC",
    "Ohm",
    "kOhm",
    "S",
    "mS",
    "uS",
    "F",
    "uF",
    "nF",
    "pF",
    "H",
    "uH",
    "nH",
    "s",
    "ms",
    "us",
    "ns",
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    "W",
    "J",
    "N",
    "Pa",
    "Wb",
    "T",
    "Bq",
    "lm",
    "lx",
    "Gy",
    "Sv",
    "kat",
]
