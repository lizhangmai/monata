import numpy as np
import pytest

from monata.measure.spec import SpecTable
from monata.measure.summary import AnalysisSummary, ac_summary, noise_summary, tran_summary
from monata.sim.results import AnalysisResult, Waveform
from monata.sim.results import SimResult
from monata.units import Quantity, quantity


def _lowpass(freq, f3db=1e3, dc_gain_db=20.0):
    gain_linear = 10 ** (dc_gain_db / 20)
    return gain_linear / (1 + 1j * freq / f3db)


def test_analysis_summary_is_lookup_and_json_safe():
    summary = AnalysisSummary(
        "ac",
        {"bandwidth": np.float64(1e6), "missing": None},
        units={"bandwidth": "Hz"},
        metadata={"array": np.array([1, 2])},
    )

    assert summary["bandwidth"] == 1e6
    assert summary.value("bandwidth") == 1e6
    with pytest.raises(ValueError, match="ac.missing"):
        summary.value("missing")
    assert summary.reasons == {}
    assert summary.to_dict()["metadata"]["array"] == [1, 2]


def test_analysis_summary_value_with_unit_preserves_registered_units():
    summary = AnalysisSummary("ac", {"bandwidth": 2.0}, units={"bandwidth": "MHz"})

    typed = summary.value_with_unit("bandwidth")

    assert isinstance(typed, Quantity)
    assert typed.unit.symbol == "MHz"
    assert typed.to("Hz").value == pytest.approx(2e6)
    assert summary.value("bandwidth") == 2.0


def test_analysis_summary_value_with_unit_keeps_display_units_numeric():
    summary = AnalysisSummary("ac", {"gain": 42.0}, units={"gain": "dB"})

    assert summary.value_with_unit("gain") == 42.0


def test_ac_summary_builds_gain_bandwidth_and_margins_from_complex_waveform():
    freq = np.logspace(0, 7, 4000)
    response = _lowpass(freq)
    result = SimResult(
        status="ok",
        waveforms={"out": response},
        sweep_var=freq,
        corner=None,
        metadata={"analysis": "ac"},
    )

    summary = ac_summary(result, "out")

    assert summary.analysis == "ac"
    assert pytest.approx(summary.value("gain"), abs=0.1) == 20.0
    assert pytest.approx(summary.value("bandwidth"), rel=0.05) == 1e3
    assert pytest.approx(summary.value("unity_gain_freq"), rel=0.1) == 1e4
    assert pytest.approx(summary.value("phase_margin"), abs=1.0) == 95.7
    assert summary.value("gain_margin") == float("inf")
    assert summary.units["bandwidth"] == "Hz"


def test_ac_summary_uses_waveform_bound_frequency_axis():
    freq = np.logspace(0, 7, 4000)
    response = Waveform.from_array(
        "out",
        _lowpass(freq),
        abscissa=freq,
        abscissa_name="frequency",
    )
    result = AnalysisResult("ac", {"out": response})

    summary = ac_summary(result, "out")

    assert pytest.approx(summary.value("bandwidth"), rel=0.05) == 1e3
    assert summary.units["unity_gain_freq"] == "Hz"


def test_ac_summary_converts_unit_frequency_abscissa():
    freq_khz = np.logspace(-3, 4, 4000)
    response = _lowpass(freq_khz * 1e3)
    frequency = Waveform.from_unit_array(
        "frequency",
        quantity(freq_khz, "kHz"),
        quantity="frequency",
    )
    result = AnalysisResult("ac", {"out": Waveform("out", response)}, abscissa=frequency)

    summary = ac_summary(result, "out")

    assert pytest.approx(summary.value("bandwidth"), rel=0.05) == 1e3
    assert summary.units["bandwidth"] == "Hz"


def test_ac_summary_preserves_explicit_phase_waveform_units():
    freq = Waveform.from_unit_array(
        "frequency",
        quantity(np.array([1.0, 10.0, 100.0, 1000.0]), "Hz"),
        quantity="frequency",
    )
    gain_db = Waveform("gain", np.array([10.0, 5.0, -5.0, -20.0]), quantity="gain")
    phase = Waveform.from_unit_array(
        "phase",
        quantity(np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2]), "rad"),
        quantity="phase",
    )
    result = AnalysisResult("ac", {"gain": gain_db, "phase": phase}, abscissa=freq)

    summary = ac_summary(result, "gain", phase_waveform="phase")

    assert summary.value("phase_margin") == pytest.approx(90.0)
    assert summary.value("gain_margin") == float("inf")
    assert summary.units["phase_margin"] == "deg"


def test_tran_summary_returns_common_time_domain_metrics():
    time = np.linspace(0, 10e-6, 10000)
    out = 1.0 - np.exp(-time / 1e-6)
    result = SimResult(
        status="ok",
        waveforms={"out": out},
        sweep_var=time,
        corner=None,
        metadata={"analysis": "tran"},
    )

    summary = tran_summary(result, "out")

    assert pytest.approx(summary.value("rise_time"), rel=0.05) == 2.2e-6
    assert summary.value("slew_rate") > 0
    assert summary.value("peak_to_peak") > 0.99
    assert summary["fall_time"] is None
    assert "Crossing" in summary.reasons["fall_time"]


def test_tran_summary_uses_waveform_bound_time_axis_and_units():
    time = np.array([0.0, 1e-9, 2e-9, 3e-9, 4e-9])
    out = Waveform.from_array(
        "out",
        np.array([0.0, 0.1, 0.9, 1.0, 1.0]),
        unit="V",
        quantity="voltage",
        abscissa=time,
        abscissa_name="time",
    )
    result = AnalysisResult("tran", {"out": out})

    summary = tran_summary(result, "out")

    assert summary.value("peak_to_peak") == pytest.approx(1.0)
    assert summary.units["peak_to_peak"] == "V"
    assert summary.units["slew_rate"] == "V/s"


def test_tran_summary_preserves_unit_waveform_and_abscissa_metrics():
    time = Waveform.from_unit_array(
        "time",
        quantity(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), "ms"),
        quantity="time",
    )
    out = Waveform.from_unit_array(
        "out",
        quantity(np.array([0.0, 100.0, 900.0, 1000.0, 1000.0]), "mV"),
        quantity="voltage",
    )
    result = AnalysisResult("tran", {"out": out}, abscissa=time)

    summary = tran_summary(result, "out")

    assert summary.value("rise_time") == pytest.approx(1e-3)
    assert summary.units["rise_time"] == "s"
    assert summary.value("peak_to_peak") == pytest.approx(1000.0)
    assert summary.units["peak_to_peak"] == "mV"
    peak_to_peak = summary.value_with_unit("peak_to_peak")
    assert isinstance(peak_to_peak, Quantity)
    assert peak_to_peak.to("V").value == pytest.approx(1.0)
    assert summary.units["slew_rate"] == "mV/s"


def test_tran_summary_delay_uses_optional_input_waveform():
    time = np.linspace(0, 5e-6, 10000)
    inp = np.where(time >= 1e-6, 1.0, 0.0)
    out = np.where(time >= 2e-6, 1.0, 0.0)
    result = SimResult(
        status="ok",
        waveforms={"in": inp, "out": out},
        sweep_var=time,
        corner=None,
        metadata={"analysis": "tran"},
    )

    summary = tran_summary(result, "out", input_waveform="in")

    assert pytest.approx(summary.value("delay"), rel=0.05) == 1e-6


def test_tran_summary_records_delay_failure_reason():
    time = np.linspace(0, 5e-6, 10000)
    inp = np.zeros_like(time)
    out = np.ones_like(time)
    result = SimResult(
        status="ok",
        waveforms={"in": inp, "out": out},
        sweep_var=time,
        corner=None,
        metadata={"analysis": "tran"},
    )

    summary = tran_summary(result, "out", input_waveform="in")

    assert summary["delay"] is None
    assert "delay" in summary.reasons
    assert "not found" in summary.reasons["delay"]


def test_noise_summary_uses_ngspice_noise_totals_metadata():
    result = SimResult(
        status="ok",
        waveforms={"onoise_spectrum": np.array([1e-9, 2e-9])},
        sweep_var=np.array([1.0, 10.0]),
        corner=None,
        metadata={
            "analysis": "noise",
            "noise_totals": {"onoise_total": np.float64(3e-9), "inoise_total": np.float64(4e-12)},
        },
    )

    summary = noise_summary(result)

    assert summary.value("onoise_total") == 3e-9
    assert summary.value("inoise_total") == 4e-12
    assert summary.units["onoise_total"] == "V"
    assert summary.units["inoise_total"] == "A"


def test_spec_summary_reads_analysis_summary_values():
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=None,
        corner=None,
        summaries={"ac": AnalysisSummary("ac", {"bandwidth": 2e6}, units={"bandwidth": "Hz"})},
    )
    table = SpecTable()
    table.add_summary("bandwidth", "ac", "bandwidth", min=1e6, unit="Hz")

    rows = table.evaluate_rows([result])

    assert rows[0]["value"] == 2e6
    assert rows[0]["passed"] is True
    assert rows[0]["source"] == "summary"


def test_spec_summary_converts_analysis_summary_units():
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=None,
        corner=None,
        summaries={"ac": AnalysisSummary("ac", {"bandwidth": 2.0}, units={"bandwidth": "MHz"})},
    )
    table = SpecTable()
    table.add_summary("bandwidth", "ac", "bandwidth", min=quantity(1500.0, "kHz"), unit="kHz")

    rows = table.evaluate_rows([result])

    assert rows[0]["value"] == pytest.approx(2000.0)
    assert rows[0]["margin"] == pytest.approx(500.0)
    assert rows[0]["passed"] is True
    assert rows[0]["unit"] == "kHz"
