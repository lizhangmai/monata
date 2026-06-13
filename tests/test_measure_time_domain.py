import numpy as np
import pytest
from monata.measure.time_domain import (
    rise_time, fall_time, delay, slew_rate, overshoot,
    settling_time, cross, period, duty_cycle, peak_to_peak,
)
from monata.units import Quantity, quantity, unit


def _step_waveform(t, tau=1e-6):
    return 1.0 - np.exp(-t / tau)


class TestRiseTime:
    def test_rc_step(self):
        t = np.linspace(0, 10e-6, 10000)
        v = _step_waveform(t, tau=1e-6)
        tr = rise_time(t, v, low=0.1, high=0.9)
        assert pytest.approx(tr, rel=0.05) == 2.2e-6

    def test_custom_thresholds(self):
        t = np.linspace(0, 10e-6, 10000)
        v = _step_waveform(t, tau=1e-6)
        tr = rise_time(t, v, low=0.2, high=0.8)
        assert pytest.approx(tr, rel=0.05) == 1.39e-6

    def test_unit_array_time_axis_preserves_time_outputs(self):
        t = quantity(np.array([0.0, 1.0]), "ms")
        v = np.array([0.0, 1.0])

        tr = rise_time(t, v, low=0.1, high=0.9)
        crossing = cross(t, v, threshold=0.5)
        slope = slew_rate(t, v)

        assert isinstance(tr, Quantity)
        assert tr.unit is unit("s")
        assert tr.to("ms").value == pytest.approx(0.8)
        assert isinstance(crossing, Quantity)
        assert crossing.unit is unit("s")
        assert crossing.to("ms").value == pytest.approx(0.5)
        assert isinstance(slope, Quantity)
        assert slope.unit.compatible_with(unit("Hz"))
        assert slope.to("Hz").value == pytest.approx(1e3)

    def test_time_axis_requires_time_compatible_units(self):
        with pytest.raises(ValueError, match="time-compatible"):
            rise_time(quantity(np.array([0.0, 1.0]), "V"), np.array([0.0, 1.0]))

    def test_rejects_invalid_fraction_bounds(self):
        with pytest.raises(ValueError, match="rise_time low must be between 0 and 1"):
            rise_time(np.array([0.0, 1.0]), np.array([0.0, 1.0]), low=-0.1)
        with pytest.raises(ValueError, match="rise_time low must be less than high"):
            rise_time(np.array([0.0, 1.0]), np.array([0.0, 1.0]), low=0.9, high=0.1)


class TestFallTime:
    def test_rc_decay(self):
        t = np.linspace(0, 10e-6, 10000)
        v = np.exp(-t / 1e-6)
        tf = fall_time(t, v, low=0.1, high=0.9)
        assert pytest.approx(tf, rel=0.05) == 2.2e-6

    def test_preserves_time_axis_units(self):
        t = quantity(np.array([0.0, 1.0]), "ms")
        v = np.array([1.0, 0.0])

        tf = fall_time(t, v, low=0.1, high=0.9)

        assert isinstance(tf, Quantity)
        assert tf.unit is unit("s")
        assert tf.to("ms").value == pytest.approx(0.8)

    def test_rejects_invalid_fraction_bounds(self):
        with pytest.raises(ValueError, match="fall_time high must be between 0 and 1"):
            fall_time(np.array([0.0, 1.0]), np.array([1.0, 0.0]), high=1.1)


class TestDelay:
    def test_propagation_delay(self):
        t = np.linspace(0, 10e-6, 10000)
        tau = 1e-6
        v_in = np.where(t > 1e-6, 1.0, 0.0)
        v_out = np.where(t > 1e-6, 1.0 - np.exp(-(t - 1e-6) / tau), 0.0)
        d = delay(t, v_in, v_out, threshold=0.5)
        assert pytest.approx(d, rel=0.1) == 0.693e-6

    def test_accepts_independent_input_and_output_edges(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v_in = np.array([0.0, 1.0, 1.0, 1.0])
        v_out = np.array([1.0, 1.0, 0.0, 0.0])

        d = delay(t, v_in, v_out, threshold=0.5, input_edge="rising", output_edge="falling")

        assert d == pytest.approx(1.0)

    def test_accepts_crossing_indices(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        v_in = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        v_out = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        d = delay(t, v_in, v_out, threshold=0.5, input_n=2, output_n=2)

        assert d == pytest.approx(1.0)

    def test_preserves_time_axis_units(self):
        t = quantity(np.array([0.0, 1.0, 2.0]), "ms")
        v_in = np.array([0.0, 1.0, 1.0])
        v_out = np.array([0.0, 0.0, 1.0])

        d = delay(t, v_in, v_out, threshold=0.5)

        assert isinstance(d, Quantity)
        assert d.unit is unit("s")
        assert d.to("ms").value == pytest.approx(1.0)

    def test_validates_edge_index(self):
        with pytest.raises(ValueError, match="n must be positive"):
            delay(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), input_n=0)

    def test_rejects_invalid_threshold_fraction(self):
        with pytest.raises(ValueError, match="delay threshold must be between 0 and 1"):
            delay(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), threshold=1.2)


class TestSlewRate:
    def test_linear_ramp(self):
        t = np.linspace(0, 1e-6, 1000)
        v = t * 1e6
        sr = slew_rate(t, v)
        assert pytest.approx(sr, rel=0.01) == 1e6

    def test_preserves_waveform_units_per_second(self):
        t = np.array([0.0, 1e-3, 2e-3])
        v = quantity(np.array([0.0, 1.0, 2.0]), "mV")

        sr = slew_rate(t, v)

        assert isinstance(sr, Quantity)
        assert sr.unit.compatible_with(unit("V") / unit("s"))
        assert sr.value == pytest.approx(1e3)
        assert sr.to(unit("V") / unit("s")).value == pytest.approx(1.0)


class TestOvershoot:
    def test_underdamped(self):
        t = np.linspace(0, 10e-6, 10000)
        wn = 2 * np.pi * 1e6
        zeta = 0.3
        wd = wn * np.sqrt(1 - zeta**2)
        v = 1.0 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(wd * t))
        os = overshoot(t, v, final_value=1.0)
        assert os > 0.3

    def test_final_value_converts_to_waveform_units(self):
        t = np.array([0.0, 1.0, 2.0])
        v = quantity(np.array([0.0, 1100.0, 1000.0]), "mV")

        os = overshoot(t, v, final_value=quantity(1.0, "V"))

        assert os == pytest.approx(0.1)


class TestSettlingTime:
    def test_rc_settling(self):
        t = np.linspace(0, 20e-6, 20000)
        v = _step_waveform(t, tau=1e-6)
        ts = settling_time(t, v, final_value=1.0, tolerance=0.01)
        assert pytest.approx(ts, rel=0.1) == 4.6e-6

    def test_interpolates_entry_into_settling_band(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 0.5, 0.98, 1.0])
        ts = settling_time(t, v, final_value=1.0, tolerance=0.1)
        assert ts == pytest.approx(1.0 + (0.9 - 0.5) / (0.98 - 0.5))

    def test_returns_last_sample_when_waveform_never_settles(self):
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([0.0, 0.2, 0.5])
        ts = settling_time(t, v, final_value=1.0, tolerance=0.1)
        assert ts == pytest.approx(2.0)

    def test_preserves_time_axis_units(self):
        t = quantity(np.array([0.0, 1.0, 2.0, 3.0]), "ms")
        v = np.array([0.0, 0.5, 0.98, 1.0])

        ts = settling_time(t, v, final_value=1.0, tolerance=0.1)

        assert isinstance(ts, Quantity)
        assert ts.unit is unit("s")
        assert ts.to("ms").value == pytest.approx(1.0 + (0.9 - 0.5) / (0.98 - 0.5))

    def test_preserves_time_axis_units_when_already_settled(self):
        t = quantity(np.array([0.0, 1.0]), "ms")
        v = np.array([1.0, 1.0])

        ts = settling_time(t, v, final_value=1.0, tolerance=0.1)

        assert isinstance(ts, Quantity)
        assert ts.unit is unit("s")
        assert ts.to("ms").value == pytest.approx(0.0)

    def test_final_value_converts_to_waveform_units(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = quantity(np.array([0.0, 500.0, 980.0, 1000.0]), "mV")

        ts = settling_time(t, v, final_value=quantity(1.0, "V"), tolerance=0.1)

        assert ts == pytest.approx(1.0 + (900.0 - 500.0) / (980.0 - 500.0))

    def test_rejects_negative_tolerance(self):
        with pytest.raises(ValueError, match="settling tolerance must be non-negative"):
            settling_time(np.array([0.0, 1.0]), np.array([0.0, 1.0]), tolerance=-0.1)


class TestCross:
    def test_rising_cross(self):
        t = np.linspace(0, 10e-6, 10000)
        v = _step_waveform(t, tau=1e-6)
        tc = cross(t, v, threshold=0.5, edge="rising", n=1)
        assert pytest.approx(tc, rel=0.05) == 0.693e-6

    def test_falling_and_either_crossings(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 0.0, 1.0])

        assert cross(t, v, threshold=0.5, edge="falling") == pytest.approx(1.5)
        assert cross(t, v, threshold=0.5, edge="either", n=1) == pytest.approx(0.5)
        assert cross(t, v, threshold=0.5, edge="either", n=2) == pytest.approx(1.5)
        assert cross(t, v, threshold=0.5, edge="either", n=3) == pytest.approx(2.5)

    def test_validates_edge_index_and_time_axis(self):
        with pytest.raises(ValueError, match="edge must"):
            cross(np.array([0.0, 1.0]), np.array([0.0, 1.0]), threshold=0.5, edge="both")
        with pytest.raises(ValueError, match="n must be positive"):
            cross(np.array([0.0, 1.0]), np.array([0.0, 1.0]), threshold=0.5, n=0)
        with pytest.raises(ValueError, match="strictly increasing"):
            cross(np.array([0.0, 0.0]), np.array([0.0, 1.0]), threshold=0.5)

    def test_threshold_converts_to_waveform_units(self):
        t = np.array([0.0, 1.0, 2.0])
        v = quantity(np.array([0.0, 1000.0, 0.0]), "mV")

        crossing = cross(t, v, threshold=quantity(0.5, "V"))

        assert crossing == pytest.approx(0.5)

    def test_accepts_time_window(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 0.0, 1.0])

        crossing = cross(t, v, threshold=0.5, edge="rising", start=1.0, stop=3.0)

        assert crossing == pytest.approx(2.5)

    def test_rejects_invalid_time_window(self):
        with pytest.raises(ValueError, match="crossing start must be <= stop"):
            cross(np.array([0.0, 1.0]), np.array([0.0, 1.0]), threshold=0.5, start=1.0, stop=0.0)

    def test_threshold_requires_value_compatible_units(self):
        t = np.array([0.0, 1.0])
        v = quantity(np.array([0.0, 1.0]), "V")

        with pytest.raises(ValueError, match="value-compatible"):
            cross(t, v, threshold=quantity(1.0, "s"))


class TestPeriod:
    def test_sine_period(self):
        freq = 1e6
        t = np.linspace(0, 10e-6, 100000)
        v = np.sin(2 * np.pi * freq * t)
        p = period(t, v, threshold=0.0)
        assert pytest.approx(p, rel=0.02) == 1e-6

    def test_preserves_time_axis_units(self):
        t = quantity(np.array([0.0, 1.0, 2.0, 3.0]), "ms")
        v = np.array([0.0, 1.0, 0.0, 1.0])

        p = period(t, v, threshold=0.5)

        assert isinstance(p, Quantity)
        assert p.unit is unit("s")
        assert p.to("ms").value == pytest.approx(2.0)

    def test_threshold_converts_to_waveform_units(self):
        t = quantity(np.array([0.0, 1.0, 2.0, 3.0]), "ms")
        v = quantity(np.array([0.0, 1000.0, 0.0, 1000.0]), "mV")

        p = period(t, v, threshold=quantity(0.5, "V"))

        assert isinstance(p, Quantity)
        assert p.to("ms").value == pytest.approx(2.0)


class TestDutyCycle:
    def test_50_percent(self):
        t = np.linspace(0, 10e-6, 100000)
        v = np.sin(2 * np.pi * 1e6 * t)
        dc = duty_cycle(t, v, threshold=0.0)
        assert pytest.approx(dc, rel=0.02) == 0.5

    def test_nonuniform_samples_are_time_weighted(self):
        t = np.array([0.0, 1.0, 2.0, 4.0])
        v = np.array([1.0, 1.0, 0.0, 0.0])
        dc = duty_cycle(t, v, threshold=0.5)
        assert dc == pytest.approx(1.5 / 4.0)

    def test_threshold_converts_to_waveform_units(self):
        t = np.array([0.0, 1.0, 2.0, 4.0])
        v = quantity(np.array([1000.0, 1000.0, 0.0, 0.0]), "mV")

        dc = duty_cycle(t, v, threshold=quantity(0.5, "V"))

        assert dc == pytest.approx(1.5 / 4.0)

    def test_requires_strictly_increasing_time_values(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            duty_cycle(np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0]))


class TestPeakToPeak:
    def test_sine(self):
        t = np.linspace(0, 10e-6, 10000)
        v = 2.5 * np.sin(2 * np.pi * 1e6 * t)
        pp = peak_to_peak(t, v)
        assert pytest.approx(pp, rel=0.01) == 5.0

    def test_preserves_waveform_units(self):
        t = np.array([0.0, 1.0, 2.0])
        v = quantity(np.array([-500.0, 250.0, 1000.0]), "mV")

        pp = peak_to_peak(t, v)

        assert isinstance(pp, Quantity)
        assert pp.unit is unit("mV")
        assert pp.value == pytest.approx(1500.0)
        assert pp.to("V").value == pytest.approx(1.5)
