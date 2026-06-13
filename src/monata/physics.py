"""Small circuit physics helpers used by schematic and sizing workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BOLTZMANN_CONSTANT = 1.380649e-23
ELEMENTARY_CHARGE = 1.602176634e-19
KELVIN_OFFSET = 273.15
SPEED_OF_LIGHT = 299_792_458.0
VACUUM_PERMEABILITY = 4.0 * np.pi * 1e-7
VACUUM_PERMITTIVITY = 1.0 / (VACUUM_PERMEABILITY * SPEED_OF_LIGHT**2)
AVOGADRO_CONSTANT = 6.02214076e23


def kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin."""

    return float(celsius) + KELVIN_OFFSET


def celsius(kelvin_value: float) -> float:
    """Convert Kelvin to Celsius."""

    return _positive_temperature_kelvin(kelvin_value) - KELVIN_OFFSET


def temperature(celsius: float | None = None, *, kelvin: float | None = None) -> float:
    """Return an absolute temperature in Kelvin from Celsius or Kelvin input."""

    if celsius is not None and kelvin is not None:
        raise ValueError("provide either celsius or kelvin, not both")
    if celsius is not None:
        value = float(celsius) + KELVIN_OFFSET
    elif kelvin is not None:
        value = _finite(kelvin, "kelvin")
    else:
        raise ValueError("temperature requires celsius or kelvin")
    return _positive_temperature_kelvin(value)


def thermal_energy(temperature_celsius: float | None = None, *, kelvin: float | None = None) -> float:
    """Return kT in joules for a Celsius or Kelvin temperature."""

    celsius_value = 27.0 if temperature_celsius is None and kelvin is None else temperature_celsius
    return BOLTZMANN_CONSTANT * temperature(celsius=celsius_value, kelvin=kelvin)


def thermal_voltage(temperature_celsius: float | None = None, *, kelvin: float | None = None) -> float:
    """Return kT/q in volts for a Celsius or Kelvin temperature."""

    return thermal_energy(temperature_celsius, kelvin=kelvin) / ELEMENTARY_CHARGE


def conductor_resistance(*, resistivity: float, length: float, area: float) -> float:
    """Return conductor resistance from bulk resistivity and geometry."""

    return _positive(resistivity, "resistivity") * _positive(length, "length") / _positive(area, "area")


@dataclass(frozen=True)
class ConductiveMaterial:
    """Electrical material properties for conductor sizing helpers."""

    name: str
    resistivity: float
    reference_temperature_celsius: float = 20.0
    temperature_coefficient: float = 0.0
    atomic_number: int | None = None
    atomic_mass: float | None = None
    density: float | None = None
    thermal_conductivity: float | None = None
    electron_mobility: float | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("material name is required")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "resistivity", _positive(self.resistivity, "resistivity"))
        reference = _finite(self.reference_temperature_celsius, "reference_temperature_celsius")
        _positive_temperature_kelvin(kelvin(reference))
        object.__setattr__(self, "reference_temperature_celsius", reference)
        object.__setattr__(
            self,
            "temperature_coefficient",
            _finite(self.temperature_coefficient, "temperature_coefficient"),
        )
        object.__setattr__(self, "atomic_number", _optional_positive_int(self.atomic_number, "atomic_number"))
        object.__setattr__(self, "atomic_mass", _optional_positive(self.atomic_mass, "atomic_mass"))
        object.__setattr__(self, "density", _optional_positive(self.density, "density"))
        object.__setattr__(
            self,
            "thermal_conductivity",
            _optional_positive(self.thermal_conductivity, "thermal_conductivity"),
        )
        object.__setattr__(
            self,
            "electron_mobility",
            _optional_finite(self.electron_mobility, "electron_mobility"),
        )

    def resistivity_at(self, temperature_celsius: float | None = None) -> float:
        """Return resistivity adjusted by a linear temperature coefficient."""

        temperature = self.reference_temperature_celsius if temperature_celsius is None else temperature_celsius
        temperature_value = _finite(temperature, "temperature_celsius")
        _positive_temperature_kelvin(kelvin(temperature_value))
        factor = 1 + self.temperature_coefficient * (temperature_value - self.reference_temperature_celsius)
        if factor <= 0:
            raise ValueError("temperature correction must keep resistivity positive")
        return self.resistivity * factor

    def conductor_resistance(
        self,
        *,
        length: float,
        area: float,
        temperature_celsius: float | None = None,
    ) -> float:
        """Return conductor resistance using this material at the requested temperature."""

        return conductor_resistance(
            resistivity=self.resistivity_at(temperature_celsius),
            length=length,
            area=area,
        )


@dataclass(frozen=True)
class ShockleyDiode:
    """Numerical Shockley diode model for sizing and post-processing helpers."""

    saturation_current: float = 10e-12
    ideality_factor: float = 1.0
    temperature_celsius: float = 25.0

    def __post_init__(self) -> None:
        _positive(self.saturation_current, "saturation_current")
        _positive(self.ideality_factor, "ideality_factor")
        _positive_temperature_kelvin(kelvin(self.temperature_celsius))

    @property
    def temperature_kelvin(self) -> float:
        return kelvin(self.temperature_celsius)

    @property
    def thermal_voltage(self) -> float:
        return BOLTZMANN_CONSTANT * self.temperature_kelvin / ELEMENTARY_CHARGE

    def current(self, voltage: Any) -> float | np.ndarray:
        """Return diode current for a scalar or array-like diode voltage."""

        voltage_array = np.asarray(voltage)
        current = self.saturation_current * (
            np.exp(voltage_array / (self.ideality_factor * self.thermal_voltage)) - 1
        )
        return _scalar_or_array(current)

    def dynamic_resistance(self, voltage: Any) -> float | np.ndarray:
        """Return small-signal resistance dV/dI at the requested voltage."""

        current = np.asarray(self.current(voltage))
        resistance = self.ideality_factor * self.thermal_voltage / (current + self.saturation_current)
        return _scalar_or_array(resistance)


def sheet_resistor_value(
    *,
    length: float,
    width: float,
    sheet_resistance: float,
    contact_resistance: float = 0.0,
    contacts: int = 0,
) -> float:
    """Return a rectangular resistor value from sheet resistance geometry."""

    length_value = _positive(length, "length")
    width_value = _positive(width, "width")
    sheet_value = _positive(sheet_resistance, "sheet_resistance")
    contact_value = _nonnegative(contact_resistance, "contact_resistance")
    contact_count = _nonnegative_int(contacts, "contacts")
    return sheet_value * (length_value / width_value) + contact_count * contact_value


def sheet_resistor_length(
    *,
    resistance: float,
    width: float,
    sheet_resistance: float,
    contact_resistance: float = 0.0,
    contacts: int = 0,
) -> float:
    """Return the required rectangular resistor length for a target value."""

    resistance_value = _positive(resistance, "resistance")
    width_value = _positive(width, "width")
    sheet_value = _positive(sheet_resistance, "sheet_resistance")
    contact_value = _nonnegative(contact_resistance, "contact_resistance")
    contact_count = _nonnegative_int(contacts, "contacts")
    channel_resistance = resistance_value - contact_count * contact_value
    if channel_resistance <= 0:
        raise ValueError("resistance must exceed total contact resistance")
    return channel_resistance * width_value / sheet_value


def _positive_temperature_kelvin(value: float) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError("temperature must be above absolute zero")
    return number


def _finite(value: float, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive(value: float, label: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _optional_finite(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label)


def _optional_positive(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    return _positive(value, label)


def _optional_positive_int(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    number = int(value)
    if number != value or number <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _nonnegative(value: float, label: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _nonnegative_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    number = int(value)
    if number != value or number < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    if value.ndim == 0:
        return float(value)
    return value


COPPER = ConductiveMaterial(
    name="copper",
    atomic_number=29,
    atomic_mass=63.546e-3,
    density=8.96e3,
    thermal_conductivity=401,
    resistivity=16.78e-9,
    reference_temperature_celsius=20.0,
    temperature_coefficient=0.00393,
    electron_mobility=-4.6e3,
)


__all__ = [
    "AVOGADRO_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "COPPER",
    "ConductiveMaterial",
    "ELEMENTARY_CHARGE",
    "KELVIN_OFFSET",
    "ShockleyDiode",
    "SPEED_OF_LIGHT",
    "VACUUM_PERMEABILITY",
    "VACUUM_PERMITTIVITY",
    "celsius",
    "conductor_resistance",
    "kelvin",
    "sheet_resistor_length",
    "sheet_resistor_value",
    "temperature",
    "thermal_energy",
    "thermal_voltage",
]
