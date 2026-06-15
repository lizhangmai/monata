import numpy as np
import pytest
from monata.measure.statistics import histogram, sigma_yield, worst_case, sensitivity
from monata.units import Quantity, UnitArray, quantity, unit


class TestHistogram:
    def test_basic(self):
        rng = np.random.default_rng(42)
        values = rng.normal(1.0, 0.1, 1000)
        bin_edges, counts = histogram(values, bins=20)
        assert len(bin_edges) == 21
        assert len(counts) == 20
        assert counts.sum() == 1000

    def test_custom_bins(self):
        values = np.array([1, 2, 3, 4, 5], dtype=float)
        bin_edges, counts = histogram(values, bins=5)
        assert len(counts) == 5

    def test_preserves_value_units_on_bin_edges(self):
        values = quantity(np.array([0.0, 500.0, 1000.0]), "mV")

        bin_edges, counts = histogram(values, bins=2)

        assert isinstance(bin_edges, UnitArray)
        assert bin_edges.unit is unit("mV")
        np.testing.assert_allclose(bin_edges.values, np.array([0.0, 500.0, 1000.0]))
        np.testing.assert_array_equal(counts, np.array([1, 2]))

    def test_rejects_empty_or_nonfinite_values(self):
        with pytest.raises(ValueError, match="not be empty"):
            histogram(np.array([]))
        with pytest.raises(ValueError, match="finite"):
            histogram(np.array([1.0, np.nan]))


class TestSigmaYield:
    def test_all_pass(self):
        rng = np.random.default_rng(42)
        values = rng.normal(1.0, 0.01, 1000)
        y = sigma_yield(values, spec_min=0.5, spec_max=1.5)
        assert y == pytest.approx(1.0, abs=0.01)

    def test_partial_fail(self):
        values = np.array([0.9, 1.0, 1.1, 1.2, 1.5, 2.0])
        y = sigma_yield(values, spec_min=0.8, spec_max=1.3)
        assert y == pytest.approx(4 / 6, abs=0.01)

    def test_min_only(self):
        values = np.array([0.5, 1.0, 1.5, 2.0])
        y = sigma_yield(values, spec_min=0.8)
        assert y == pytest.approx(3 / 4)

    def test_max_only(self):
        values = np.array([0.5, 1.0, 1.5, 2.0])
        y = sigma_yield(values, spec_max=1.2)
        assert y == pytest.approx(2 / 4)

    def test_specs_convert_to_value_units(self):
        values = quantity(np.array([500.0, 1000.0, 1500.0, 2000.0]), "mV")

        y = sigma_yield(values, spec_min=quantity(0.8, "V"), spec_max=quantity(1.2, "V"))

        assert y == pytest.approx(1 / 4)

    def test_rejects_invalid_spec_range(self):
        with pytest.raises(ValueError, match="less than or equal"):
            sigma_yield(np.array([1.0, 2.0]), spec_min=2.0, spec_max=1.0)

    def test_rejects_empty_values(self):
        with pytest.raises(ValueError, match="not be empty"):
            sigma_yield(np.array([]), spec_min=0.0)

    def test_rejects_incompatible_spec_units(self):
        values = quantity(np.array([1.0, 2.0]), "V")

        with pytest.raises(ValueError, match="value-compatible"):
            sigma_yield(values, spec_min=quantity(1.0, "s"))


class TestWorstCase:
    def test_basic(self):
        values = np.array([1.0, 2.0, 0.5, 3.0, -1.0])
        wc_min, wc_max = worst_case(values)
        assert wc_min == -1.0
        assert wc_max == 3.0

    def test_preserves_value_units(self):
        values = quantity(np.array([1000.0, -500.0, 250.0]), "mV")

        wc_min, wc_max = worst_case(values)

        assert isinstance(wc_min, Quantity)
        assert isinstance(wc_max, Quantity)
        assert wc_min.unit is unit("mV")
        assert wc_min.value == pytest.approx(-500.0)
        assert wc_max.to("V").value == pytest.approx(1.0)

    def test_rejects_empty_values(self):
        with pytest.raises(ValueError, match="not be empty"):
            worst_case(np.array([]))


class TestSensitivity:
    def test_linear(self):
        param_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metric_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        s = sensitivity(param_values, metric_values)
        assert pytest.approx(s, rel=0.01) == 10.0

    def test_preserves_metric_per_parameter_units(self):
        param_values = quantity(np.array([1.0, 2.0, 3.0]), "kOhm")
        metric_values = quantity(np.array([10.0, 20.0, 30.0]), "mV")

        s = sensitivity(param_values, metric_values)

        assert isinstance(s, Quantity)
        assert s.unit.compatible_with(unit("A"))
        assert s.value == pytest.approx(10.0)
        assert s.to("A").value == pytest.approx(1e-5)

    def test_single_sample_preserves_sensitivity_units(self):
        s = sensitivity(quantity(np.array([1.0]), "kOhm"), quantity(np.array([10.0]), "mV"))

        assert isinstance(s, Quantity)
        assert s.unit.compatible_with(unit("A"))
        assert s.to("A").value == pytest.approx(0.0)

    def test_constant(self):
        param_values = np.array([1.0, 2.0, 3.0])
        metric_values = np.array([5.0, 5.0, 5.0])
        s = sensitivity(param_values, metric_values)
        assert pytest.approx(s, abs=0.01) == 0.0

    def test_single_sample_has_zero_sensitivity(self):
        assert sensitivity(np.array([1.0]), np.array([5.0])) == 0.0

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            sensitivity(np.array([1.0, 2.0]), np.array([3.0]))

    def test_rejects_nonfinite_values(self):
        with pytest.raises(ValueError, match="finite"):
            sensitivity(np.array([1.0, np.inf]), np.array([3.0, 4.0]))
