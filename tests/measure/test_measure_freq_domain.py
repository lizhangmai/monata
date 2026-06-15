import numpy as np
import pytest
from monata.measure.freq_domain import (
    BodeTrace, bode_trace,
    gain, bandwidth, gain_bandwidth_product, phase_margin,
    gain_margin, unity_gain_freq, rejection_at_freq,
)
from monata.units import Quantity, quantity, unit


def _first_order_lowpass(freq, f3db=1e6, dc_gain_db=60.0):
    """First-order lowpass: H(f) = A0 / (1 + j*f/f3db)."""
    a0 = 10 ** (dc_gain_db / 20)
    h = a0 / (1 + 1j * freq / f3db)
    mag_db = 20 * np.log10(np.abs(h))
    phase_deg = np.angle(h, deg=True)
    return mag_db, phase_deg


def test_bode_trace_projects_complex_response_to_gain_and_phase():
    frequency = np.array([1.0, 10.0, 100.0])
    response = np.array([1.0 + 0.0j, 0.0 - 1.0j, 10.0 + 0.0j])

    trace = bode_trace(frequency, response)

    assert isinstance(trace, BodeTrace)
    np.testing.assert_allclose(trace.frequency, frequency)
    np.testing.assert_allclose(trace.gain_db, np.array([0.0, 0.0, 20.0]))
    np.testing.assert_allclose(trace.phase, np.array([0.0, -90.0, 0.0]))
    assert trace.phase_unit == "deg"


def test_bode_trace_can_return_radians_and_unwrap_phase():
    frequency = np.array([1.0, 10.0])
    response = np.exp(1j * np.deg2rad(np.array([170.0, -170.0])))

    wrapped = bode_trace(frequency, response, phase_unit="rad")
    unwrapped = bode_trace(frequency, response, phase_unit="rad", unwrap_phase=True)

    np.testing.assert_allclose(wrapped.phase, np.deg2rad(np.array([170.0, -170.0])))
    np.testing.assert_allclose(unwrapped.phase, np.deg2rad(np.array([170.0, 190.0])))
    assert wrapped.phase_unit == "rad"


def test_bode_trace_converts_unit_array_frequency_to_hz():
    frequency = quantity(np.array([1.0, 10.0]), "kHz")
    response = np.array([1.0 + 0.0j, 0.0 - 1.0j])

    trace = bode_trace(frequency, response)

    np.testing.assert_allclose(trace.frequency, np.array([1e3, 1e4]))
    np.testing.assert_allclose(trace.phase, np.array([0.0, -90.0]))


def test_bode_trace_exports_column_arrays_without_copy_by_default():
    frequency = np.array([1.0, 10.0, 100.0])
    response = np.array([1.0 + 0.0j, 0.0 - 1.0j, 10.0 + 0.0j])

    trace = bode_trace(frequency, response)
    columns = trace.to_arrays()

    assert list(columns) == ["frequency", "gain_db", "phase_deg"]
    np.testing.assert_allclose(columns["frequency"], frequency)
    np.testing.assert_allclose(columns["gain_db"], np.array([0.0, 0.0, 20.0]))
    np.testing.assert_allclose(columns["phase_deg"], np.array([0.0, -90.0, 0.0]))
    assert np.shares_memory(columns["frequency"], trace.frequency)
    assert np.shares_memory(columns["gain_db"], trace.gain_db)
    assert np.shares_memory(columns["phase_deg"], trace.phase)


def test_bode_trace_exports_phase_unit_column_and_copy():
    frequency = np.array([1.0, 10.0])
    response = np.exp(1j * np.deg2rad(np.array([170.0, -170.0])))

    trace = bode_trace(frequency, response, phase_unit="rad", unwrap_phase=True)
    columns = trace.to_arrays(copy=True)

    assert list(columns) == ["frequency", "gain_db", "phase_rad"]
    np.testing.assert_allclose(columns["phase_rad"], np.deg2rad(np.array([170.0, 190.0])))
    assert not np.shares_memory(columns["frequency"], trace.frequency)
    assert not np.shares_memory(columns["gain_db"], trace.gain_db)
    assert not np.shares_memory(columns["phase_rad"], trace.phase)


def test_bode_trace_to_arrays_can_convert_phase_units():
    frequency = np.array([1.0, 10.0])
    response = np.exp(1j * np.deg2rad(np.array([90.0, -45.0])))

    degree_trace = bode_trace(frequency, response)
    radian_columns = degree_trace.to_arrays(phase_unit="rad")
    radian_trace = bode_trace(frequency, response, phase_unit="rad")
    degree_columns = radian_trace.to_arrays(phase_unit="deg")

    assert list(radian_columns) == ["frequency", "gain_db", "phase_rad"]
    np.testing.assert_allclose(radian_columns["phase_rad"], np.deg2rad(np.array([90.0, -45.0])))
    assert not np.shares_memory(radian_columns["phase_rad"], degree_trace.phase)
    assert list(degree_columns) == ["frequency", "gain_db", "phase_deg"]
    np.testing.assert_allclose(degree_columns["phase_deg"], np.array([90.0, -45.0]))
    assert not np.shares_memory(degree_columns["phase_deg"], radian_trace.phase)


def test_bode_trace_to_arrays_rejects_unknown_phase_unit():
    trace = bode_trace(np.array([1.0]), np.array([1.0 + 0.0j]))

    with pytest.raises(ValueError, match="phase_unit"):
        trace.to_arrays(phase_unit="turn")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("frequency", "response", "message"),
    [
        ([0.0, 1.0], [1.0, 1.0], "positive"),
        ([10.0, 1.0], [1.0, 1.0], "strictly increasing"),
        ([1.0], [1.0, 2.0], "same shape"),
        ([], [], "at least one point"),
        ([[1.0]], [[1.0]], "one-dimensional"),
        (quantity(np.array([1.0]), "V"), [1.0], "frequency-compatible"),
    ],
)
def test_bode_trace_validates_inputs(frequency, response, message):
    with pytest.raises(ValueError, match=message):
        bode_trace(frequency, response)


def test_frequency_measurements_convert_unit_array_frequency_to_hz():
    frequency = quantity(np.array([1.0, 10.0, 1000.0]), "kHz")
    mag_db = np.array([20.0, 0.0, -20.0])

    ugf = unity_gain_freq(frequency, mag_db)

    assert isinstance(ugf, Quantity)
    assert ugf.unit is unit("Hz")
    assert ugf.value == pytest.approx(1e4)
    assert rejection_at_freq(frequency, mag_db, quantity(100.0, "kHz")) == pytest.approx(-10.0)


class TestGain:
    def test_dc_gain(self):
        freq = np.logspace(0, 9, 1000)
        mag_db, _ = _first_order_lowpass(freq, dc_gain_db=60.0)
        g = gain(freq, mag_db)
        assert pytest.approx(g, abs=0.1) == 60.0


class TestBandwidth:
    def test_3db_bandwidth(self):
        freq = np.logspace(0, 9, 10000)
        mag_db, _ = _first_order_lowpass(freq, f3db=1e6, dc_gain_db=60.0)
        bw = bandwidth(freq, mag_db, gain_drop=-3.0)
        assert pytest.approx(bw, rel=0.05) == 1e6

    def test_preserves_frequency_axis_units(self):
        freq = quantity(np.logspace(0, 3, 10000), "kHz")
        mag_db, _ = _first_order_lowpass(freq.to("Hz").values, f3db=1e6, dc_gain_db=60.0)

        bw = bandwidth(freq, mag_db, gain_drop=-3.0)

        assert isinstance(bw, Quantity)
        assert bw.unit is unit("Hz")
        assert bw.to("MHz").value == pytest.approx(1.0, rel=0.05)


class TestGBW:
    def test_gbw(self):
        freq = np.logspace(0, 9, 10000)
        mag_db, _ = _first_order_lowpass(freq, f3db=1e6, dc_gain_db=60.0)
        gbw = gain_bandwidth_product(freq, mag_db)
        assert pytest.approx(gbw, rel=0.1) == 1e9

    def test_preserves_frequency_axis_units(self):
        freq = quantity(np.logspace(0, 3, 10000), "kHz")
        mag_db, _ = _first_order_lowpass(freq.to("Hz").values, f3db=1e6, dc_gain_db=60.0)

        gbw = gain_bandwidth_product(freq, mag_db)

        assert isinstance(gbw, Quantity)
        assert gbw.unit is unit("Hz")
        assert gbw.to("GHz").value == pytest.approx(1.0, rel=0.1)


class TestPhaseMargin:
    def test_first_order(self):
        freq = np.logspace(0, 9, 10000)
        mag_db, phase_deg = _first_order_lowpass(freq, f3db=1e6, dc_gain_db=60.0)
        pm = phase_margin(freq, mag_db, phase_deg)
        assert pytest.approx(pm, abs=5) == 90.0

    def test_accepts_angle_unit_array_phase(self):
        freq = np.logspace(0, 9, 10000)
        mag_db, phase_deg = _first_order_lowpass(freq, f3db=1e6, dc_gain_db=60.0)
        phase_rad = quantity(np.deg2rad(phase_deg), "rad")

        pm = phase_margin(freq, mag_db, phase_rad)

        assert pytest.approx(pm, abs=5) == 90.0

    def test_requires_angle_compatible_phase_units(self):
        freq = np.array([1.0, 10.0])
        mag_db = np.array([1.0, -1.0])
        phase = quantity(np.array([-90.0, -180.0]), "Hz")

        with pytest.raises(ValueError, match="angle-compatible"):
            phase_margin(freq, mag_db, phase)


class TestGainMargin:
    def test_first_order_infinite(self):
        freq = np.logspace(0, 9, 10000)
        mag_db, phase_deg = _first_order_lowpass(freq, f3db=1e6, dc_gain_db=60.0)
        gm = gain_margin(freq, mag_db, phase_deg)
        assert gm == float("inf")

    def test_accepts_angle_unit_array_phase(self):
        freq = np.array([1.0, 10.0, 100.0])
        mag_db = np.array([20.0, 10.0, -6.0])
        phase = quantity(np.deg2rad(np.array([-90.0, -180.0, -270.0])), "rad")

        gm = gain_margin(freq, mag_db, phase)

        assert gm == pytest.approx(-10.0)


class TestUnityGainFreq:
    def test_ugf(self):
        freq = np.logspace(0, 9, 10000)
        mag_db, _ = _first_order_lowpass(freq, f3db=1e6, dc_gain_db=60.0)
        ugf = unity_gain_freq(freq, mag_db)
        assert pytest.approx(ugf, rel=0.1) == 1e9

    def test_accepts_rising_gain_crossing(self):
        freq = np.array([1.0, 10.0, 100.0])
        mag_db = np.array([-20.0, 0.0, 20.0])

        ugf = unity_gain_freq(freq, mag_db)

        assert ugf == pytest.approx(10.0)

    def test_interpolates_rising_gain_crossing(self):
        freq = np.array([1.0, 100.0])
        mag_db = np.array([-20.0, 20.0])

        ugf = unity_gain_freq(freq, mag_db)

        assert ugf == pytest.approx(10.0)

    def test_preserves_frequency_axis_units(self):
        freq = quantity(np.logspace(0, 6, 10000), "kHz")
        mag_db, _ = _first_order_lowpass(freq.to("Hz").values, f3db=1e6, dc_gain_db=60.0)

        ugf = unity_gain_freq(freq, mag_db)

        assert isinstance(ugf, Quantity)
        assert ugf.unit is unit("Hz")
        assert ugf.to("GHz").value == pytest.approx(1.0, rel=0.1)


class TestRejection:
    def test_rejection_at_1ghz(self):
        freq = np.logspace(0, 9, 10000)
        mag_db, _ = _first_order_lowpass(freq, f3db=1e6, dc_gain_db=60.0)
        rej = rejection_at_freq(freq, mag_db, f_target=1e9)
        assert pytest.approx(rej, abs=1) == 0.0

    def test_rejects_invalid_target_frequency(self):
        freq = np.array([1.0, 10.0, 100.0])
        mag_db = np.array([0.0, -20.0, -40.0])

        with pytest.raises(ValueError, match="target frequency must be positive"):
            rejection_at_freq(freq, mag_db, f_target=0.0)
        with pytest.raises(ValueError, match="target frequency must be finite"):
            rejection_at_freq(freq, mag_db, f_target=float("nan"))
        with pytest.raises(ValueError, match="within sampled frequency range"):
            rejection_at_freq(freq, mag_db, f_target=1000.0)
        with pytest.raises(ValueError, match="within sampled frequency range"):
            rejection_at_freq(freq, mag_db, f_target=quantity(0.5, "Hz"))

    def test_rejects_non_finite_frequency_axis(self):
        with pytest.raises(ValueError, match="frequency values must be finite"):
            rejection_at_freq(np.array([1.0, np.nan, 100.0]), np.array([0.0, -20.0, -40.0]), f_target=10.0)
