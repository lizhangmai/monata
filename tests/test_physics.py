import pytest
import numpy as np

from monata.physics import (
    AVOGADRO_CONSTANT,
    BOLTZMANN_CONSTANT,
    COPPER,
    ConductiveMaterial,
    ShockleyDiode,
    SPEED_OF_LIGHT,
    VACUUM_PERMEABILITY,
    VACUUM_PERMITTIVITY,
    celsius,
    conductor_resistance,
    kelvin,
    sheet_resistor_length,
    sheet_resistor_value,
    temperature,
    thermal_energy,
    thermal_voltage,
)


def test_physical_constants_and_temperature_selection():
    assert SPEED_OF_LIGHT == pytest.approx(299_792_458.0)
    assert AVOGADRO_CONSTANT == pytest.approx(6.02214076e23)
    assert VACUUM_PERMEABILITY == pytest.approx(4 * np.pi * 1e-7)
    assert VACUUM_PERMITTIVITY == pytest.approx(1 / (VACUUM_PERMEABILITY * SPEED_OF_LIGHT**2))
    assert kelvin(27) == pytest.approx(300.15)
    assert celsius(300.15) == pytest.approx(27)
    assert temperature(celsius=27) == pytest.approx(300.15)
    assert temperature(kelvin=300.15) == pytest.approx(300.15)
    with pytest.raises(ValueError, match="either celsius or kelvin"):
        temperature(celsius=27, kelvin=300.15)
    with pytest.raises(ValueError, match="requires celsius or kelvin"):
        temperature()
    with pytest.raises(ValueError, match="above absolute zero"):
        temperature(kelvin=0)


def test_thermal_voltage_at_room_temperature():
    assert thermal_energy() == pytest.approx(BOLTZMANN_CONSTANT * 300.15)
    assert thermal_voltage() == pytest.approx(0.025865, rel=1e-4)
    assert thermal_energy(27) == pytest.approx(BOLTZMANN_CONSTANT * 300.15)
    assert thermal_voltage(27) == pytest.approx(0.025865, rel=1e-4)
    assert thermal_energy(kelvin=300.15) == pytest.approx(BOLTZMANN_CONSTANT * 300.15)
    assert thermal_voltage(kelvin=300.15) == pytest.approx(thermal_voltage(27))
    with pytest.raises(ValueError, match="either celsius or kelvin"):
        thermal_energy(27, kelvin=300.15)


def test_conductor_resistance_from_bulk_resistivity_and_geometry():
    assert conductor_resistance(resistivity=1.68e-8, length=10e-3, area=1e-6) == pytest.approx(1.68e-4)


def test_conductive_material_temperature_adjusted_resistivity_and_resistance():
    assert COPPER.name == "copper"
    assert COPPER.atomic_number == 29
    assert COPPER.density == pytest.approx(8.96e3)
    assert COPPER.resistivity_at() == pytest.approx(16.78e-9)
    assert COPPER.resistivity_at(20) == pytest.approx(16.78e-9)
    assert COPPER.resistivity_at(70) == pytest.approx(16.78e-9 * (1 + 0.00393 * 50))
    assert COPPER.conductor_resistance(length=10e-3, area=1e-6, temperature_celsius=70) == pytest.approx(
        COPPER.resistivity_at(70) * 10e-3 / 1e-6
    )


def test_conductive_material_accepts_custom_materials():
    material = ConductiveMaterial(
        "aluminum interconnect",
        resistivity=2.65e-8,
        reference_temperature_celsius=20,
        temperature_coefficient=0.00429,
    )

    assert material.resistivity_at(30) == pytest.approx(2.65e-8 * (1 + 0.00429 * 10))


def test_sheet_resistor_value_and_inverse_length():
    resistance = sheet_resistor_value(
        length=10e-6,
        width=2e-6,
        sheet_resistance=100,
        contact_resistance=5,
        contacts=2,
    )
    length = sheet_resistor_length(
        resistance=resistance,
        width=2e-6,
        sheet_resistance=100,
        contact_resistance=5,
        contacts=2,
    )

    assert resistance == pytest.approx(510)
    assert length == pytest.approx(10e-6)


def test_sheet_resistor_helpers_validate_geometry():
    with pytest.raises(ValueError, match="width must be positive"):
        sheet_resistor_value(length=1, width=0, sheet_resistance=100)
    with pytest.raises(ValueError, match="exceed total contact"):
        sheet_resistor_length(resistance=5, width=1, sheet_resistance=100, contact_resistance=5, contacts=2)


def test_shockley_diode_current_and_dynamic_resistance():
    diode = ShockleyDiode(saturation_current=1e-12, ideality_factor=2, temperature_celsius=27)

    assert diode.temperature_kelvin == pytest.approx(300.15)
    assert diode.thermal_voltage == pytest.approx(thermal_voltage(27))
    assert diode.current(0.0) == pytest.approx(0.0)
    assert diode.current(0.7) == pytest.approx(1e-12 * (np.exp(0.7 / (2 * thermal_voltage(27))) - 1))
    assert diode.dynamic_resistance(0.0) == pytest.approx(2 * thermal_voltage(27) / 1e-12)


def test_shockley_diode_accepts_numpy_voltage_arrays():
    diode = ShockleyDiode(saturation_current=1e-12, ideality_factor=1, temperature_celsius=27)
    voltages = np.array([0.0, 0.1, 0.2])

    currents = diode.current(voltages)
    resistances = diode.dynamic_resistance(voltages)

    assert isinstance(currents, np.ndarray)
    assert isinstance(resistances, np.ndarray)
    assert currents.shape == voltages.shape
    assert resistances.shape == voltages.shape
    np.testing.assert_allclose(currents, 1e-12 * (np.exp(voltages / thermal_voltage(27)) - 1))


def test_physics_helpers_validate_physical_inputs():
    with pytest.raises(ValueError, match="area must be positive"):
        conductor_resistance(resistivity=1.0, length=1.0, area=0.0)
    with pytest.raises(ValueError, match="material name is required"):
        ConductiveMaterial("", resistivity=1.0)
    with pytest.raises(ValueError, match="atomic_number must be a positive integer"):
        ConductiveMaterial("bad atomic", resistivity=1.0, atomic_number=0)
    with pytest.raises(ValueError, match="temperature correction"):
        ConductiveMaterial("negative alpha", resistivity=1.0, temperature_coefficient=-1).resistivity_at(22)
    with pytest.raises(ValueError, match="saturation_current must be positive"):
        ShockleyDiode(saturation_current=0.0)
    with pytest.raises(ValueError, match="temperature must be above absolute zero"):
        ShockleyDiode(temperature_celsius=-273.15)
