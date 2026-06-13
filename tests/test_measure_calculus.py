from fractions import Fraction

import numpy as np
import pytest

from monata.measure import (
    exact_finite_difference_coefficients,
    finite_difference_coefficients,
    finite_difference_derivative,
    finite_difference_stencil,
    simple_derivative,
)
from monata.measure.calculus import cumulative_trapezoidal_integral, trapezoidal_integral
from monata.units import UnitArray, quantity, unit


def test_finite_difference_coefficients_match_common_centered_stencils():
    first = finite_difference_coefficients(1, [-1.0, 0.0, 1.0])
    second = finite_difference_coefficients(2, [-1.0, 0.0, 1.0])

    np.testing.assert_allclose(first, np.array([-0.5, 0.0, 0.5]))
    np.testing.assert_allclose(second, np.array([1.0, -2.0, 1.0]))


def test_exact_finite_difference_coefficients_return_rational_stencils():
    first = exact_finite_difference_coefficients(1, [-1, 0, 1])
    second = exact_finite_difference_coefficients(2, [-1, 0, 1])

    assert first == (Fraction(-1, 2), Fraction(0), Fraction(1, 2))
    assert second == (Fraction(1), Fraction(-2), Fraction(1))


def test_exact_finite_difference_coefficients_support_shifted_evaluation_offset():
    forward = exact_finite_difference_coefficients(1, [0, 1, 2])
    centered_at_one = exact_finite_difference_coefficients(1, [0, 1, 2], evaluation_offset=1)

    assert forward == (Fraction(-3, 2), Fraction(2), Fraction(-1, 2))
    assert centered_at_one == (Fraction(-1, 2), Fraction(0), Fraction(1, 2))


def test_finite_difference_stencil_builds_named_uniform_grid_stencils():
    centered_offsets, centered_weights = finite_difference_stencil(1, 2)
    forward_offsets, forward_weights = finite_difference_stencil(1, 2, side="forward")
    backward_offsets, backward_weights = finite_difference_stencil(1, 2, side="backward")

    assert centered_offsets == (-1, 0, 1)
    assert forward_offsets == (0, 1, 2)
    assert backward_offsets == (0, -1, -2)
    np.testing.assert_allclose(centered_weights, np.array([-0.5, 0.0, 0.5]))
    np.testing.assert_allclose(forward_weights, np.array([-1.5, 2.0, -0.5]))
    np.testing.assert_allclose(backward_weights, np.array([1.5, -2.0, 0.5]))


def test_finite_difference_stencil_can_return_exact_rational_weights():
    offsets, weights = finite_difference_stencil(1, 2, side="centred", exact=True)

    assert offsets == (-1, 0, 1)
    assert weights == (Fraction(-1, 2), Fraction(0), Fraction(1, 2))


def test_simple_derivative_returns_adjacent_sample_slopes():
    x = np.array([0.0, 0.5, 2.0])
    y = np.array([0.0, 1.0, 4.0])

    sample_x, slope = simple_derivative(x, y)

    np.testing.assert_allclose(sample_x, np.array([0.0, 0.5]))
    np.testing.assert_allclose(slope, np.array([2.0, 2.0]))


def test_simple_derivative_preserves_unit_array_axes_and_slopes():
    x = quantity(np.array([0.0, 0.5, 2.0]), "ms")
    y = quantity(np.array([0.0, 1.0, 4.0]), "mV")

    sample_x, slope = simple_derivative(x, y)

    assert isinstance(sample_x, UnitArray)
    assert sample_x.unit is unit("ms")
    np.testing.assert_allclose(sample_x.values, np.array([0.0, 0.5]))
    assert isinstance(slope, UnitArray)
    assert slope.unit.compatible_with(unit("V") / unit("s"))
    np.testing.assert_allclose(slope.values, np.array([2.0, 2.0]))
    np.testing.assert_allclose(slope.to(unit("V") / unit("s")).values, np.array([2.0, 2.0]))


def test_finite_difference_derivative_is_exact_for_polynomial_inside_stencil_order():
    x = np.linspace(-1.0, 1.0, 21)
    y = x**3 - 2 * x**2 + x

    derivative = finite_difference_derivative(x, y, derivative_order=1, accuracy_order=4)

    np.testing.assert_allclose(derivative, 3 * x**2 - 4 * x + 1, atol=1e-11)


def test_finite_difference_derivative_supports_nonuniform_samples():
    x = np.array([0.0, 0.1, 0.4, 0.9, 1.5, 2.2, 3.0])
    y = x**2

    derivative = finite_difference_derivative(x, y, derivative_order=1, accuracy_order=2)

    np.testing.assert_allclose(derivative, 2 * x, atol=1e-12)


def test_finite_difference_derivative_preserves_complex_waveforms():
    x = np.linspace(0.0, 1.0, 11)
    y = (1.0 + 2.0j) * x**2

    derivative = finite_difference_derivative(x, y, derivative_order=1, accuracy_order=2)

    assert np.iscomplexobj(derivative)
    np.testing.assert_allclose(derivative, 2 * (1.0 + 2.0j) * x, atol=1e-12)


def test_finite_difference_derivative_preserves_unit_array_units():
    x = quantity(np.linspace(0.0, 2.0, 9), "ms")
    y = quantity(2.0 * x.values + 1.0, "mV")

    derivative = finite_difference_derivative(x, y, derivative_order=1, accuracy_order=2)

    assert isinstance(derivative, UnitArray)
    assert derivative.unit.compatible_with(unit("V") / unit("s"))
    np.testing.assert_allclose(derivative.values, np.full(x.shape, 2.0), atol=1e-12)
    np.testing.assert_allclose(derivative.to(unit("V") / unit("s")).values, np.full(x.shape, 2.0), atol=1e-12)


def test_finite_difference_derivative_applies_x_units_to_dimensionless_values():
    x = quantity(np.linspace(0.0, 2.0, 9), "ms")
    y = 3.0 * x.values

    derivative = finite_difference_derivative(x, y, derivative_order=1, accuracy_order=2)

    assert isinstance(derivative, UnitArray)
    assert derivative.unit.compatible_with(unit("Hz"))
    np.testing.assert_allclose(derivative.values, np.full(x.shape, 3.0), atol=1e-12)
    np.testing.assert_allclose(derivative.to("Hz").values, np.full(x.shape, 3000.0), atol=1e-12)


def test_trapezoidal_integral_returns_cumulative_and_definite_values():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])

    cumulative = cumulative_trapezoidal_integral(x, y)
    shifted = cumulative_trapezoidal_integral(x, y, initial=1.0)
    definite = trapezoidal_integral(x, y)

    np.testing.assert_allclose(cumulative, np.array([0.0, 0.5, 3.0]))
    np.testing.assert_allclose(shifted, np.array([1.0, 1.5, 4.0]))
    assert definite == pytest.approx(3.0)


def test_trapezoidal_integral_preserves_unit_array_units():
    x = quantity(np.array([0.0, 1.0, 2.0]), "ms")
    y = quantity(np.array([0.0, 1.0, 4.0]), "mV")

    cumulative = cumulative_trapezoidal_integral(x, y, initial=quantity(1.0, unit("mV") * unit("ms")))
    definite = trapezoidal_integral(x, y)

    assert isinstance(cumulative, UnitArray)
    assert cumulative.unit.compatible_with(unit("V") * unit("s"))
    np.testing.assert_allclose(cumulative.values, np.array([1.0, 1.5, 4.0]))
    np.testing.assert_allclose(cumulative.to(unit("V") * unit("s")).values, np.array([1.0e-6, 1.5e-6, 4.0e-6]))
    assert definite.unit.compatible_with(unit("V") * unit("s"))
    assert definite.value == pytest.approx(3.0)
    assert definite.to(unit("V") * unit("s")).value == pytest.approx(3.0e-6)


def test_finite_difference_helpers_validate_inputs():
    with pytest.raises(ValueError, match="positive integer"):
        finite_difference_coefficients(0, [-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="more points"):
        exact_finite_difference_coefficients(2, [0, 1])
    with pytest.raises(ValueError, match="unique"):
        exact_finite_difference_coefficients(1, [0, 0, 1])
    with pytest.raises(ValueError, match="finite"):
        exact_finite_difference_coefficients(1, [0, float("nan"), 1])
    with pytest.raises(ValueError, match="side"):
        finite_difference_stencil(1, 2, side="middle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="even"):
        finite_difference_stencil(1, 3)
    with pytest.raises(ValueError, match="more points"):
        finite_difference_stencil(4, 2)
    with pytest.raises(ValueError, match="unique"):
        finite_difference_coefficients(1, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="same shape"):
        simple_derivative([0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        finite_difference_derivative([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="not enough samples"):
        finite_difference_derivative([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], accuracy_order=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        cumulative_trapezoidal_integral([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="initial integral value must be scalar"):
        cumulative_trapezoidal_integral([0.0, 1.0], [0.0, 1.0], initial=[0.0, 1.0])
    with pytest.raises(ValueError, match="integral-compatible"):
        cumulative_trapezoidal_integral(quantity([0.0, 1.0], "ms"), quantity([0.0, 1.0], "mV"), initial=quantity(1.0, "V"))
