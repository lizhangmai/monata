from typing import Any

import numpy as np
import pytest

import monata.units as units
from monata.units import (
    A,
    Bq,
    C,
    Degree,
    F,
    Gy,
    Hz,
    J,
    K,
    N,
    Ohm,
    Pa,
    Quantity,
    S,
    Sv,
    T,
    UnitError,
    UnitArray,
    V,
    W,
    Wb,
    amplitude_to_rms,
    as_unit,
    atto,
    cd,
    deca,
    deg,
    exa,
    femto,
    format_spice_number,
    format_spice_value,
    format_spice_values,
    giga,
    hecto,
    join_spice_values,
    kOhm,
    kat,
    kg,
    kilo,
    lm,
    lx,
    m,
    mS,
    mega,
    micro,
    milli,
    mol,
    nano,
    parse_spice_number,
    peta,
    pico,
    quantity,
    rad,
    rms_to_amplitude,
    scaled_unit,
    si_prefix,
    sr,
    tera,
    yocto,
    yotta,
    zepto,
    zetta,
    mA,
    mV,
    mm,
    ms,
    s,
    uF,
    unit,
    unit_value,
)


def test_parse_spice_number_suffixes():
    assert parse_spice_number("1k") == 1000.0
    assert parse_spice_number("1kHz") == 1000.0
    assert parse_spice_number("2meg") == 2_000_000.0
    assert parse_spice_number("2MegOhm") == 2_000_000.0
    assert parse_spice_number("1mil") == pytest.approx(25.4e-6)
    assert parse_spice_number("3m") == 0.003
    assert parse_spice_number("3ms") == 0.003
    assert parse_spice_number("4u") == 4e-6
    assert parse_spice_number("5a") == 5e-18
    assert parse_spice_number("5aF") == 5e-18
    assert parse_spice_number("10V") == 10.0
    assert parse_spice_number("10A") == 10.0
    assert parse_spice_number(5) == 5.0


def test_parse_spice_number_ltspice_rkm_dialect_is_explicit():
    with pytest.raises(UnitError, match="invalid SPICE number"):
        parse_spice_number("2k3")

    assert parse_spice_number("2k3", dialect="ltspice") == pytest.approx(2300.0)
    assert parse_spice_number("4R7", dialect="ltspice") == pytest.approx(4.7)
    assert parse_spice_number("1u5", dialect="ltspice") == pytest.approx(1.5e-6)

    with pytest.raises(UnitError, match="unknown SPICE number dialect"):
        parse_spice_number("1k", dialect="bad")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (0, "0"),
        (1000, "1k"),
        (2_000_000, "2meg"),
        (0.003, "3m"),
        (4e-6, "4u"),
        (5e-18, "5a"),
    ],
)
def test_format_spice_number_uses_suffixes(value, rendered):
    assert format_spice_number(value) == rendered


def test_format_spice_value_renders_scalar_tokens():
    assert format_spice_value(quantity(1200, "mV")) == "1.2"
    assert format_spice_value(1.0) == "1.0"
    assert format_spice_value("V(in)*2") == "V(in)*2"
    assert format_spice_value(None) == ""
    assert format_spice_value(None, none="0") == "0"


def test_units_do_not_expose_inspice_style_str_spice_helpers():
    assert not hasattr(units, "str_spice")
    assert not hasattr(units, "str_spice_list")


def test_quantity_exposes_monata_string_methods():
    voltage = quantity(1200, "mV")
    resistance = 1 @ kOhm

    assert isinstance(voltage, Quantity)
    assert isinstance(resistance, Quantity)
    assert voltage.spice() == "1.2"
    assert voltage.str(spice=True) == "1.2"
    assert voltage.str(unit=False) == "1200"
    assert voltage.str() == "1200mV"
    assert voltage.str_space() == "1200 mV"
    assert resistance.str() == "1kOhm"
    assert not hasattr(voltage, "str_spice")


def test_format_spice_values_renders_multiple_scalar_tokens():
    assert format_spice_values(0 @ V, "raw", None) == ("0", "raw", "")
    assert format_spice_values(None, 1 @ kOhm, none="0") == ("0", "1k")


def test_join_spice_values_joins_non_empty_scalar_tokens():
    assert join_spice_values(0 @ V, "", None, 1 @ kOhm, "raw") == "0 1k raw"
    assert join_spice_values(None, 1 @ kOhm, none="0") == "0 1k"
    assert join_spice_values(1, 2, 3, sep=",") == "1,2,3"


def test_format_spice_value_rejects_unit_arrays():
    with pytest.raises(UnitError, match="expects a scalar value"):
        format_spice_value(quantity(np.array([1.0, 2.0]), V))


def test_si_prefix_lookup():
    assert si_prefix("k") == 1e3
    assert si_prefix("meg") == 1e6
    assert si_prefix("M") == 1e6
    assert si_prefix("m") == 1e-3
    assert si_prefix("u") == 1e-6
    assert si_prefix("μ") == 1e-6
    assert si_prefix("h") == 1e2
    assert si_prefix("da") == 1e1


def test_common_circuit_units_are_registered_and_compatible_with_derived_forms():
    assert unit("volt") is V
    assert unit("kelvin") is K
    assert unit("Degree") is Degree
    assert unit("°C") is Degree
    assert unit("degC") is Degree
    assert unit("meter") is m
    assert unit("mm") is mm
    assert unit("Ω") is Ohm
    assert unit("rad") is rad
    assert unit("deg") is deg

    assert (A / V) is S
    assert (A * s) is C
    assert (V * A) is W
    assert (V * s) is Wb
    assert (Wb / m.square()) is T

    assert quantity(1, "mm").to(m).value == pytest.approx(1e-3)
    assert quantity(180, deg).to(rad).value == pytest.approx(np.pi)


def test_si_derived_units_are_registered_and_reduce_from_monata_dimensions():
    assert unit("steradian") is sr
    assert unit("kilogram") is kg
    assert unit("mole") is mol
    assert unit("candela") is cd
    assert unit("joule") is J
    assert unit("newton") is N
    assert unit("pascal") is Pa
    assert unit("becquerel") is Bq
    assert unit("gray") is Gy
    assert unit("sievert") is Sv
    assert unit("katal") is kat

    assert (W * s) is J
    assert (J / s) is W
    assert (J / m) is N
    assert (N / m.square()) is Pa
    assert (lm / m.square()) is lx
    assert (mol / s) is kat
    assert (J / kg) is Gy
    assert Sv.compatible_with(Gy)
    assert Bq.compatible_with(Hz)


def test_scaled_unit_generates_prefixed_units_and_preserves_spice_rendering():
    meg_ohm = scaled_unit(Ohm, "meg")
    assert meg_ohm.symbol == "MegOhm"
    assert meg_ohm.dimensions == Ohm.dimensions
    assert quantity(2, meg_ohm).spice() == "2meg"

    assert unit("uF") is uF
    assert quantity(1, uF).to(F).value == pytest.approx(1e-6)
    assert quantity(np.array([1.0, 2.0]), uF).to(F).values.tolist() == pytest.approx([1e-6, 2e-6])


def test_unit_conversion_and_compatibility():
    volts = quantity(1200, "mV")
    assert isinstance(volts, Quantity)

    converted = volts.to(V)

    assert converted.unit is V
    assert converted.value == pytest.approx(1.2)
    assert converted.compatible_with(unit("mV"))
    assert not converted.compatible_with(A)


def test_quantity_exposes_monata_clone_and_to_methods():
    volts = quantity(1200, "mV")
    base_volts = quantity(1.2, V)
    assert isinstance(volts, Quantity)
    assert isinstance(base_volts, Quantity)

    cloned = volts.clone()
    replaced = volts.clone_prefixed_unit(600)
    converted = volts.to(V)

    assert volts.prefixed_unit is mV
    assert volts.scale == pytest.approx(1e-3)
    assert volts.is_same_unit(base_volts)
    assert volts.is_same_unit("V")
    assert not volts.is_same_unit(A)
    assert volts.is_same_power(quantity(1, "mV"))
    assert not volts.is_same_power(base_volts)
    assert cloned == volts
    assert cloned is not volts
    assert cloned.unit is volts.unit
    assert replaced.unit is volts.unit
    assert replaced.value == pytest.approx(600)
    assert converted.unit is V
    assert converted.value == pytest.approx(1.2)
    assert not hasattr(volts, "convert")


def test_quantity_converts_to_prefix_power_and_canonises_registered_units():
    millivolts = quantity(1200, "mV")
    small_volts = quantity(0.0012, V)
    resistance = quantity(2200, Ohm)
    unregistered_prefix = quantity(1e-9, V)
    assert isinstance(millivolts, Quantity)
    assert isinstance(small_volts, Quantity)
    assert isinstance(resistance, Quantity)
    assert isinstance(unregistered_prefix, Quantity)

    volts = millivolts.convert_to_power(0)
    back_to_millivolts = volts.convert_to_power(-3)
    canonised_voltage = small_volts.canonise()
    canonised_resistance = resistance.canonise()

    assert millivolts.get_prefixed_unit(-3) is mV
    assert volts.unit is V
    assert volts.value == pytest.approx(1.2)
    assert back_to_millivolts.unit is mV
    assert back_to_millivolts.value == pytest.approx(1200)
    assert canonised_voltage.unit is mV
    assert canonised_voltage.value == pytest.approx(1.2)
    assert canonised_resistance.unit is kOhm
    assert canonised_resistance.value == pytest.approx(2.2)
    assert unregistered_prefix.canonise() is unregistered_prefix
    with pytest.raises(UnitError, match="no registered unit"):
        millivolts.convert_to_power(-9)


def test_incompatible_unit_conversion_has_stable_error():
    volts = quantity(1.0, V)
    assert isinstance(volts, Quantity)

    with pytest.raises(UnitError, match="incompatible units"):
        volts.to(A)


def test_unit_array_conversion_preserves_shape_and_numpy_access():
    values = quantity(np.array([1.0, 2.0, 3.0]), "kOhm")
    base_values = quantity(np.array([1000.0, 2000.0, 3000.0]), Ohm)
    assert isinstance(values, UnitArray)
    assert isinstance(base_values, UnitArray)

    converted = values.to(Ohm)

    assert values.prefixed_unit is kOhm
    assert values.scale == pytest.approx(1e3)
    assert values.is_same_unit(base_values)
    assert values.is_same_unit("Ohm")
    assert not values.is_same_unit(V)
    assert values.is_same_power(quantity(np.array([4.0]), "kOhm"))
    assert not values.is_same_power(base_values)
    assert converted.unit is Ohm
    assert converted.shape == (3,)
    np.testing.assert_allclose(np.asarray(converted), np.array([1000.0, 2000.0, 3000.0]))
    assert not hasattr(values, "convert")


def test_unit_array_converts_to_prefix_power():
    values = quantity(np.array([0.001, 0.002]), V)
    assert isinstance(values, UnitArray)

    millivolts = values.convert_to_power(-3)
    volts = millivolts.convert_to_power(0)

    assert values.get_prefixed_unit(-3) is mV
    assert millivolts.unit is mV
    np.testing.assert_allclose(millivolts.values, np.array([1.0, 2.0]))
    assert volts.unit is V
    np.testing.assert_allclose(volts.values, np.array([0.001, 0.002]))
    with pytest.raises(UnitError, match="no registered unit"):
        values.convert_to_power(-9)


def test_unit_array_exposes_monata_array_view_and_unit_conversion():
    values = mV(np.array([1000.0, 2000.0]))
    assert isinstance(values, UnitArray)

    view = values.as_array()
    view[0] = 1500.0
    copied = values.as_array(copy=True)
    copied[1] = 3000.0

    assert values.values[0] == pytest.approx(1500.0)
    assert values.values[1] == pytest.approx(2000.0)
    np.testing.assert_allclose(values.to(V).as_array(), np.array([1.5, 2.0]))
    assert not hasattr(values, "as_ndarray")


def test_unit_array_indexing_slicing_and_iteration_preserve_units():
    matrix = quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), V)
    vector = quantity(np.array([1.0, 2.0]), V)
    assert isinstance(matrix, UnitArray)
    assert isinstance(vector, UnitArray)

    scalar = matrix[0, 1]
    row = matrix[1]
    column = matrix[:, 0]
    iterated = list(vector)

    assert len(matrix) == 2
    assert isinstance(scalar, Quantity)
    assert scalar.unit is V
    assert scalar.value == pytest.approx(2.0)
    assert isinstance(row, UnitArray)
    assert row.unit is V
    np.testing.assert_allclose(np.asarray(row), np.array([3.0, 4.0]))
    assert isinstance(column, UnitArray)
    assert column.unit is V
    np.testing.assert_allclose(np.asarray(column), np.array([1.0, 3.0]))
    assert all(isinstance(item, Quantity) and item.unit is V for item in iterated)
    assert [item.value for item in iterated if isinstance(item, Quantity)] == [1.0, 2.0]


def test_unit_array_assignment_converts_compatible_units():
    values = quantity(np.array([0.0, 0.0, 0.0]), V)
    assert isinstance(values, UnitArray)

    values[0] = quantity(500, "mV")
    values[1:] = quantity(np.array([1200.0, 1800.0]), "mV")

    np.testing.assert_allclose(np.asarray(values), np.array([0.5, 1.2, 1.8]))
    with pytest.raises(UnitError, match="incompatible units"):
        values[0] = quantity(1, mA)


def test_unknown_unit_and_prefix_raise_unit_error():
    with pytest.raises(UnitError, match="unknown unit"):
        unit("parsec")
    with pytest.raises(UnitError, match="unknown SI prefix"):
        si_prefix("bad")


def test_frequency_unit_conversion():
    frequency = quantity([1.0, 2.0], "kHz")
    assert isinstance(frequency, UnitArray)

    converted = frequency.to(Hz)

    np.testing.assert_allclose(np.asarray(converted), np.array([1000.0, 2000.0]))


def test_unit_call_and_matmul_shortcuts_create_quantities():
    resistance = 1 @ kOhm
    voltage = V(1.2)

    assert isinstance(resistance, Quantity)
    assert resistance.unit is kOhm
    assert isinstance(voltage, Quantity)
    assert voltage.unit is V


def test_canonical_unit_objects_create_quantities_and_arrays():
    resistance = 1 @ kOhm
    voltage = mV(1200)
    temperature = Degree(27)
    capacitances = uF([1.0, 2.0])
    simple = unit_value(3)
    simple_values = unit_value([1.0, 2.0])

    assert resistance.spice() == "1k"
    assert voltage.to(V).value == pytest.approx(1.2)
    assert isinstance(temperature, Quantity)
    assert temperature.unit is Degree
    assert temperature.spice() == "27"
    assert isinstance(capacitances, UnitArray)
    np.testing.assert_allclose(capacitances.to(uF).values, np.array([1.0, 2.0]))
    assert isinstance(simple, Quantity)
    assert simple.unit.symbol == ""
    assert simple.value == pytest.approx(3)
    assert isinstance(simple_values, UnitArray)
    assert simple_values.unit.symbol == ""
    np.testing.assert_allclose(simple_values.values, np.array([1.0, 2.0]))
    assert Hz(50).unit is Hz
    assert s(20).unit is s
    assert (20 @ ms).frequency.to(Hz).value == pytest.approx(50)


def test_units_do_not_expose_inspice_style_unit_alias_facades():
    for name in (
        "U_mV",
        "u_mV",
        "U_kOhm",
        "u_kOhm",
        "U_Degree",
        "u_Degree",
        "Frequency",
        "Period",
        "as_V",
        "as_Ohm",
        "as_Degree",
    ):
        assert not hasattr(units, name)


def test_si_prefix_helpers_scale_scalars_arrays_and_quantities():
    assert yotta(2) == pytest.approx(2e24)
    assert zetta(2) == pytest.approx(2e21)
    assert exa(2) == pytest.approx(2e18)
    assert peta(2) == pytest.approx(2e15)
    assert tera(2) == pytest.approx(2e12)
    assert giga(2) == pytest.approx(2e9)
    assert mega(2) == pytest.approx(2e6)
    assert kilo(2) == pytest.approx(2e3)
    assert hecto(2) == pytest.approx(2e2)
    assert deca(2) == pytest.approx(2e1)
    assert milli(3) == pytest.approx(3e-3)
    assert micro(4) == pytest.approx(4e-6)
    assert pico(5) == pytest.approx(5e-12)
    assert femto(6) == pytest.approx(6e-15)
    assert atto(7) == pytest.approx(7e-18)
    assert zepto(8) == pytest.approx(8e-21)
    assert yocto(9) == pytest.approx(9e-24)
    np.testing.assert_allclose(nano([1.0, 2.0]), np.array([1e-9, 2e-9]))
    assert kilo(2 @ V).value == pytest.approx(2000)
    assert kilo(2 @ V).unit is V


def test_as_unit_validates_and_converts_values():
    voltage = as_unit(1200 @ mV, V)
    resistance = as_unit(1 @ kOhm, Ohm)
    temperature = as_unit(27 @ Degree, Degree)
    currents = as_unit([1.0, 2.0], mA)
    scalar = as_unit(3, V)

    assert isinstance(voltage, Quantity)
    assert voltage.unit is V
    assert voltage.value == pytest.approx(1.2)
    assert isinstance(resistance, Quantity)
    assert resistance.unit is Ohm
    assert resistance.value == pytest.approx(1000)
    assert isinstance(temperature, Quantity)
    assert temperature.unit is Degree
    assert temperature.value == pytest.approx(27)
    assert isinstance(currents, UnitArray)
    assert currents.unit is mA
    np.testing.assert_allclose(np.asarray(currents), np.array([1.0, 2.0]))
    assert isinstance(scalar, Quantity)
    assert scalar.unit is V
    assert scalar.value == pytest.approx(3)
    assert as_unit(None, V, none=True) is None
    assert as_unit(None, Degree, none=True) is None
    with pytest.raises(UnitError, match="incompatible units"):
        as_unit(1 @ mA, V)
    with pytest.raises(UnitError, match="incompatible units"):
        as_unit(1 @ V, Degree)


def test_quantity_arithmetic_converts_compatible_units_and_derives_products():
    voltage = quantity(1200, "mV") + quantity(0.3, "V")
    current = quantity(1, mA)
    resistance = quantity(1, kOhm)
    computed_voltage = resistance * current

    assert voltage.unit is unit("mV")
    assert voltage.to(V).value == pytest.approx(1.5)
    assert computed_voltage.unit is V
    assert computed_voltage.value == pytest.approx(1.0)


def test_quantity_scalar_value_semantics_convert_compatible_units():
    voltage = quantity(1200, "mV")
    higher_voltage = quantity(1.25, V)
    zero_voltage = quantity(0, V)
    negative_voltage = quantity(-2, V)

    assert voltage == quantity(1.2, V)
    assert voltage != quantity(1.1, V)
    assert voltage != quantity(1, mA)
    assert higher_voltage > voltage
    assert voltage <= quantity(1.2, V)
    assert voltage > 1000
    assert not zero_voltage
    assert bool(voltage)
    assert int(quantity(3.8, V)) == 3
    assert float(voltage) == pytest.approx(1200.0)
    assert (-voltage).value == pytest.approx(-1200.0)
    assert (+voltage).unit is unit("mV")
    assert abs(negative_voltage).value == pytest.approx(2.0)
    with pytest.raises(UnitError, match="incompatible units"):
        _ = voltage < quantity(1, mA)


def test_quantity_and_unit_reciprocal_derive_inverse_units():
    inverse_volt = V.reciprocal()
    reciprocal_voltage = quantity(2, V).reciprocal()
    reciprocal_resistance = (2 @ kOhm).reciprocal()

    assert inverse_volt.symbol == "1/V"
    assert inverse_volt.dimensions == {"voltage": -1}
    assert reciprocal_voltage.unit.symbol == "1/V"
    assert reciprocal_voltage.value == pytest.approx(0.5)
    assert reciprocal_resistance.unit is mS
    assert reciprocal_resistance.value == pytest.approx(0.5)


def test_quantity_and_unit_square_sqrt_power_derive_units():
    squared_volt = V.square()
    squared_voltage = quantity(3, V).square()
    powered_voltage = quantity(3, V) ** 2
    rooted_voltage = quantity(9, squared_volt).sqrt()
    bad_power: Any = 0.5

    assert squared_volt.symbol == "V^2"
    assert squared_volt.dimensions == {"voltage": 2}
    assert squared_voltage.unit.symbol == "V^2"
    assert squared_voltage.value == pytest.approx(9.0)
    assert powered_voltage.unit.symbol == "V^2"
    assert powered_voltage.value == pytest.approx(9.0)
    assert rooted_voltage.unit is V
    assert rooted_voltage.value == pytest.approx(3.0)
    with pytest.raises(UnitError, match="cannot be square-rooted"):
        quantity(3, V).sqrt()
    with pytest.raises(UnitError, match="unit powers must be integers"):
        _ = quantity(3, V) ** bad_power


def test_quantity_spice_renders_base_value_with_spice_suffix():
    assert quantity(1200, "mV").spice() == "1.2"
    assert quantity(1, kOhm).spice() == "1k"
    assert quantity(20, ms).spice() == "20m"


def test_unit_array_arithmetic_preserves_and_derives_units():
    left = quantity(np.array([1.0, 2.0]), "V")
    right = quantity(np.array([250.0, 500.0]), "mV")
    conductance_current = quantity(np.array([1.0, 2.0]), mA)
    resistance = quantity(np.array([1.0, 1.0]), kOhm)

    summed = left + right
    computed_voltage = resistance * conductance_current

    assert isinstance(summed, UnitArray)
    assert summed.unit is V
    np.testing.assert_allclose(np.asarray(summed), np.array([1.25, 2.5]))
    assert computed_voltage.unit is V
    np.testing.assert_allclose(np.asarray(computed_voltage), np.array([1.0, 2.0]))


def test_unit_array_numpy_unary_ufuncs_preserve_units():
    values = quantity(np.array([-1.0, 2.0]), V)

    negated = np.negative(values)
    absolute = np.absolute(values)

    assert isinstance(negated, UnitArray)
    assert negated.unit is V
    np.testing.assert_allclose(np.asarray(negated), np.array([1.0, -2.0]))
    assert isinstance(absolute, UnitArray)
    assert absolute.unit is V
    np.testing.assert_allclose(np.asarray(absolute), np.array([1.0, 2.0]))


def test_unit_array_numpy_add_subtract_convert_compatible_units():
    left = quantity(np.array([1.0, 2.0]), V)
    right = quantity(np.array([250.0, 500.0]), "mV")
    reference: Any = 1 @ V

    summed = np.add(left, right)
    difference = np.subtract(left, right)
    reversed_difference = np.subtract(reference, right)

    assert isinstance(summed, UnitArray)
    assert summed.unit is V
    np.testing.assert_allclose(np.asarray(summed), np.array([1.25, 2.5]))
    assert isinstance(difference, UnitArray)
    assert difference.unit is V
    np.testing.assert_allclose(np.asarray(difference), np.array([0.75, 1.5]))
    assert isinstance(reversed_difference, UnitArray)
    assert reversed_difference.unit.symbol == "mV"
    np.testing.assert_allclose(np.asarray(reversed_difference), np.array([750.0, 500.0]))


def test_unit_array_numpy_multiply_divide_derive_units():
    resistance = quantity(np.array([1.0, 2.0]), kOhm)
    current = quantity(np.array([1.0, 2.0]), mA)

    voltage = np.multiply(resistance, current)
    restored_resistance = np.divide(voltage, current)

    assert isinstance(voltage, UnitArray)
    assert voltage.unit is V
    np.testing.assert_allclose(np.asarray(voltage), np.array([1.0, 4.0]))
    assert isinstance(restored_resistance, UnitArray)
    assert restored_resistance.unit is kOhm
    np.testing.assert_allclose(np.asarray(restored_resistance), np.array([1.0, 2.0]))


def test_unit_array_reciprocal_derives_inverse_units():
    voltage = quantity(np.array([2.0, 4.0]), V)
    assert isinstance(voltage, UnitArray)

    reciprocal = voltage.reciprocal()
    ufunc_reciprocal = np.reciprocal(voltage)

    assert isinstance(reciprocal, UnitArray)
    assert reciprocal.unit.symbol == "1/V"
    np.testing.assert_allclose(np.asarray(reciprocal), np.array([0.5, 0.25]))
    assert isinstance(ufunc_reciprocal, UnitArray)
    assert ufunc_reciprocal.unit.symbol == "1/V"
    np.testing.assert_allclose(np.asarray(ufunc_reciprocal), np.array([0.5, 0.25]))


def test_unit_array_square_sqrt_power_derive_units():
    voltage = quantity(np.array([2.0, 3.0]), V)
    assert isinstance(voltage, UnitArray)

    squared = voltage.square()
    ufunc_squared = np.square(voltage)
    powered = np.power(voltage, 2)
    rooted = np.sqrt(squared)

    assert isinstance(squared, UnitArray)
    assert squared.unit.symbol == "V^2"
    np.testing.assert_allclose(np.asarray(squared), np.array([4.0, 9.0]))
    assert isinstance(ufunc_squared, UnitArray)
    assert ufunc_squared.unit.symbol == "V^2"
    np.testing.assert_allclose(np.asarray(ufunc_squared), np.array([4.0, 9.0]))
    assert isinstance(powered, UnitArray)
    assert powered.unit.symbol == "V^2"
    np.testing.assert_allclose(np.asarray(powered), np.array([4.0, 9.0]))
    assert isinstance(rooted, UnitArray)
    assert rooted.unit is V
    np.testing.assert_allclose(np.asarray(rooted), np.array([2.0, 3.0]))
    with pytest.raises(UnitError, match="cannot be square-rooted"):
        np.sqrt(voltage)


def test_unit_array_numpy_comparison_and_extrema_convert_units():
    left = quantity(np.array([1.0, 2.0]), V)
    right = quantity(np.array([500.0, 2500.0]), "mV")

    greater = np.greater(left, right)
    maximum = np.maximum(left, right)

    np.testing.assert_array_equal(greater, np.array([True, False]))
    assert isinstance(maximum, UnitArray)
    assert maximum.unit is V
    np.testing.assert_allclose(np.asarray(maximum), np.array([1.0, 2.5]))


def test_unit_array_numpy_reductions_and_summaries_preserve_units():
    values = quantity(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), V)
    assert isinstance(values, UnitArray)

    total = np.sum(values)
    column_sum = np.sum(values, axis=0)
    mean = np.mean(values)
    row_maximum = np.max(values, axis=1)
    reduced_sum = np.add.reduce(values, axis=0)
    reduced_maximum = np.maximum.reduce(values, axis=1)

    assert isinstance(total, Quantity)
    assert total.unit is V
    assert total.value == pytest.approx(21.0)
    assert isinstance(column_sum, UnitArray)
    assert column_sum.unit is V
    np.testing.assert_allclose(np.asarray(column_sum), np.array([5.0, 7.0, 9.0]))
    assert isinstance(mean, Quantity)
    assert mean.unit is V
    assert mean.value == pytest.approx(3.5)
    assert isinstance(row_maximum, UnitArray)
    assert row_maximum.unit is V
    np.testing.assert_allclose(np.asarray(row_maximum), np.array([3.0, 6.0]))
    assert isinstance(reduced_sum, UnitArray)
    assert reduced_sum.unit is V
    np.testing.assert_allclose(np.asarray(reduced_sum), np.array([5.0, 7.0, 9.0]))
    assert isinstance(reduced_maximum, UnitArray)
    assert reduced_maximum.unit is V
    np.testing.assert_allclose(np.asarray(reduced_maximum), np.array([3.0, 6.0]))


def test_unit_array_summary_methods_preserve_units_and_reject_out():
    values = quantity(np.array([1.0, 2.0, 3.0]), V)
    assert isinstance(values, UnitArray)

    total = values.sum()
    mean = values.mean()
    minimum = values.min()
    maximum = values.max()

    assert isinstance(total, Quantity)
    assert total.unit is V
    assert total.value == pytest.approx(6.0)
    assert isinstance(mean, Quantity)
    assert mean.unit is V
    assert mean.value == pytest.approx(2.0)
    assert isinstance(minimum, Quantity)
    assert minimum.unit is V
    assert minimum.value == pytest.approx(1.0)
    assert isinstance(maximum, Quantity)
    assert maximum.unit is V
    assert maximum.value == pytest.approx(3.0)
    with pytest.raises(UnitError, match="do not support out"):
        values.sum(out=np.empty(1))


def test_unit_array_numpy_ufuncs_reject_incompatible_or_unsupported_units():
    voltage = quantity(np.array([1.0, 2.0]), V)
    current = quantity(np.array([1.0, 2.0]), mA)

    with pytest.raises(UnitError, match="incompatible units"):
        np.add(voltage, current)
    with pytest.raises(TypeError):
        np.sin(voltage)
    with pytest.raises(TypeError):
        np.multiply.reduce(voltage)


def test_frequency_period_and_pulsation_helpers():
    period = quantity(20, ms)
    frequency = quantity(50, Hz)

    assert period.frequency.to(Hz).value == pytest.approx(50)
    assert frequency.period.to(s).value == pytest.approx(0.02)
    assert frequency.pulsation.unit.symbol == "rad/s"
    assert frequency.pulsation.value == pytest.approx(2 * np.pi * 50)


def test_unit_array_frequency_period_and_pulsation_helpers_are_vectorized():
    periods = quantity(np.array([20.0, 10.0]), ms)
    frequencies = quantity(np.array([50.0, 100.0]), Hz)
    voltage = quantity(np.array([1.0, 2.0]), V)

    assert isinstance(periods, UnitArray)
    assert isinstance(frequencies, UnitArray)
    np.testing.assert_allclose(np.asarray(periods.frequency.to(Hz)), np.array([50.0, 100.0]))
    np.testing.assert_allclose(np.asarray(frequencies.period.to(s)), np.array([0.02, 0.01]))
    assert frequencies.pulsation.unit.symbol == "rad/s"
    np.testing.assert_allclose(np.asarray(frequencies.pulsation), 2 * np.pi * np.array([50.0, 100.0]))
    with pytest.raises(UnitError, match="cannot be converted to frequency"):
        voltage.frequency


def test_rms_and_amplitude_helpers_preserve_unit_values():
    rms_voltage = quantity(1.0, V)
    rms_current = quantity(np.array([1.0, 2.0]), mA)

    peak_voltage = rms_to_amplitude(rms_voltage)
    peak_current = rms_to_amplitude(rms_current)

    assert isinstance(peak_voltage, Quantity)
    assert peak_voltage.unit is V
    assert peak_voltage.value == pytest.approx(np.sqrt(2))
    assert isinstance(peak_current, UnitArray)
    assert peak_current.unit is mA
    np.testing.assert_allclose(np.asarray(peak_current), np.array([np.sqrt(2), 2 * np.sqrt(2)]))
    assert amplitude_to_rms(peak_voltage).value == pytest.approx(1.0)
    np.testing.assert_allclose(np.asarray(amplitude_to_rms(peak_current)), np.array([1.0, 2.0]))


def test_rms_and_amplitude_helpers_accept_plain_numeric_values():
    assert rms_to_amplitude(2.0) == pytest.approx(2.0 * np.sqrt(2))
    assert amplitude_to_rms(2.0 * np.sqrt(2)) == pytest.approx(2.0)
    np.testing.assert_allclose(rms_to_amplitude([1.0, 2.0]), np.array([np.sqrt(2), 2 * np.sqrt(2)]))
