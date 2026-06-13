
import numpy as np
import pytest

from monata.sim import result_ops
from monata.sim.results import (
    AnalysisResult,
    Waveform,
    WaveformNotFoundError,
    analysis_result_from_arrays,
)
from monata.units import Quantity, UnitArray, V, ms
pytestmark = pytest.mark.slow

def test_analysis_result_from_arrays_links_abscissa():
    time = np.array([0.0, 1e-9, 2e-9])
    values = np.array([0.0, 0.5, 1.0])

    result = analysis_result_from_arrays(
        {"out": values},
        time,
        {"analysis": "tran", "extraction": "rawfile"},
        source_vectors={"out": "v(out)"},
    )

    assert result.analysis == "tran"
    assert result.source == "rawfile"
    assert result.abscissa is not None
    assert result.abscissa.name == "time"
    assert result.waveform("out").abscissa == "time"
    assert result.waveform("out").source_vector == "v(out)"
    assert result.waveform("out").raw_vector_name == "v(out)"
    assert result.waveform("out").vector_kind == "node_voltage"
    assert result.waveform("out").quantity == "voltage"
    assert result.waveform("out").unit == "V"
    np.testing.assert_allclose(result.waveform("out").data, values)
    np.testing.assert_allclose(result.waveform("out").abscissa_data, time)
    np.testing.assert_allclose(result.waveform("out").magnitude().abscissa_data, time)


def test_analysis_result_supports_item_and_safe_attribute_lookup():
    result = AnalysisResult(
        analysis="tran",
        waveforms={
            "out": Waveform("out", np.array([1.0]), normalized_name="out_node"),
            "i(v1)": Waveform("i(v1)", np.array([0.1]), normalized_name="i_v1"),
        },
    )

    assert result["out"].name == "out"
    assert result.out_node.name == "out"
    assert result.i_v1.name == "i(v1)"


def test_analysis_result_select_waveforms_preserves_order_and_alias_lookup():
    result = AnalysisResult(
        analysis="tran",
        waveforms={
            "out": Waveform("out", np.array([1.0]), raw_vector_name="v(out)", source_vector="v(out)"),
            "i(v1)": Waveform("i(v1)", np.array([0.1]), normalized_name="i_v1"),
        },
    )

    selected = result.select_waveforms("i_v1", "v(out)", "out")

    assert list(selected) == ["i_v1", "v(out)", "out"]
    assert selected["i_v1"].name == "i(v1)"
    assert selected["v(out)"].name == "out"
    assert selected["out"].name == "out"


def test_analysis_result_supports_mapping_style_waveform_access():
    out = Waveform("out", np.array([1.0]), raw_vector_name="v(out)", source_vector="v(out)")
    branch = Waveform("i(v1)", np.array([0.1]), normalized_name="i_v1")
    result = AnalysisResult("tran", {"out": out, "i(v1)": branch})

    assert len(result) == 2
    assert list(result) == ["out", "i(v1)"]
    assert list(result.keys()) == ["out", "i(v1)"]
    assert list(result.values()) == [out, branch]
    assert list(result.items()) == [("out", out), ("i(v1)", branch)]
    assert "out" in result
    assert "v(out)" in result
    assert "i_v1" in result
    assert "missing" not in result
    assert 1 not in result
    assert result.get("v(out)") is out
    assert result.get("missing") is None
    sentinel = object()
    assert result.get("missing", sentinel) is sentinel


def test_analysis_result_lookup_accepts_case_insensitive_spice_vector_aliases():
    out = Waveform(
        "out",
        np.array([1.0]),
        raw_vector_name="v(out)",
        source_vector="v(out)",
        normalized_name="out",
    )
    supply_current = Waveform(
        "i(VDD)",
        np.array([0.1]),
        raw_vector_name="i(vdd)",
        source_vector="i(VDD)",
        normalized_name="i_vdd",
    )
    result = AnalysisResult("tran", {"out": out, "i(VDD)": supply_current})

    assert result.waveform("V(OUT)") is out
    assert result.waveform(" v(out) ") is out
    assert result["I(VDD)"] is supply_current
    assert result.waveform("i_vdd") is supply_current
    assert "V(OUT)" in result
    assert result.get("i(vdd)") is supply_current


def test_analysis_result_select_waveforms_can_ignore_missing_names():
    result = AnalysisResult(
        analysis="tran",
        waveforms={"out": Waveform("out", np.array([1.0]))},
    )

    selected = result.select_waveforms("missing", "out", missing="ignore")

    assert list(selected) == ["out"]
    with pytest.raises(ValueError, match="missing"):
        result.select_waveforms("out", missing="skip")  # type: ignore[arg-type]


def test_analysis_result_to_arrays_includes_abscissa_and_waveforms_in_order():
    result = analysis_result_from_arrays(
        {
            "out": np.array([0.0, 0.5, 1.0]),
            "in": np.array([1.0, 1.0, 0.0]),
        },
        np.array([0.0, 1e-9, 2e-9]),
        {"analysis": "tran"},
        source_vectors={"out": "v(out)", "in": "v(in)"},
    )

    arrays = result.to_arrays("v(out)", "in")

    assert list(arrays) == ["time", "v(out)", "in"]
    np.testing.assert_allclose(arrays["time"], [0.0, 1e-9, 2e-9])
    np.testing.assert_allclose(arrays["v(out)"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(arrays["in"], [1.0, 1.0, 0.0])


def test_analysis_result_to_unit_arrays_includes_abscissa_and_units_in_order():
    result = analysis_result_from_arrays(
        {
            "out": np.array([0.0, 0.5, 1.0]),
            "in": np.array([1.0, 1.0, 0.0]),
        },
        np.array([0.0, 1e-9, 2e-9]),
        {"analysis": "tran"},
        source_vectors={"out": "v(out)", "in": "v(in)"},
    )

    unit_arrays = result.to_unit_arrays("v(out)", "in")

    assert list(unit_arrays) == ["time", "v(out)", "in"]
    assert isinstance(unit_arrays["time"], UnitArray)
    assert unit_arrays["time"].unit.symbol == "s"
    assert unit_arrays["v(out)"].unit.symbol == "V"
    assert unit_arrays["in"].unit.symbol == "V"
    np.testing.assert_allclose(unit_arrays["time"].values, [0.0, 1e-9, 2e-9])
    np.testing.assert_allclose(unit_arrays["v(out)"].values, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(unit_arrays["in"].values, [1.0, 1.0, 0.0])


def test_analysis_result_to_arrays_uses_shared_waveform_abscissa_when_result_has_none():
    time = np.array([0.0, 1e-9, 2e-9])
    result = AnalysisResult(
        analysis="tran",
        waveforms={
            "out": Waveform("out", np.array([0.0, 0.5, 1.0]), unit="V", abscissa="time", abscissa_data=time),
            "in": Waveform("in", np.array([1.0, 1.0, 0.0]), unit="V", abscissa="time", abscissa_data=time),
        },
    )

    arrays = result.to_arrays("out", "in")
    unit_arrays = result.to_unit_arrays("out", "in")

    assert list(arrays) == ["time", "out", "in"]
    np.testing.assert_allclose(arrays["time"], time)
    np.testing.assert_allclose(arrays["out"], [0.0, 0.5, 1.0])
    assert list(unit_arrays) == ["time", "out", "in"]
    assert unit_arrays["time"].unit.symbol == "s"
    assert unit_arrays["out"].unit.symbol == "V"
    np.testing.assert_allclose(unit_arrays["time"].values, time)


def test_analysis_result_to_arrays_skips_unshared_waveform_abscissas():
    result = AnalysisResult(
        analysis="tran",
        waveforms={
            "out": Waveform(
                "out",
                np.array([0.0, 0.5, 1.0]),
                abscissa="time",
                abscissa_data=np.array([0.0, 1e-9, 2e-9]),
            ),
            "in": Waveform(
                "in",
                np.array([1.0, 1.0, 0.0]),
                abscissa="time",
                abscissa_data=np.array([0.0, 2e-9, 4e-9]),
            ),
        },
    )

    arrays = result.to_arrays()

    assert list(arrays) == ["out", "in"]


def test_analysis_result_to_arrays_rejects_shared_waveform_abscissa_column_conflicts():
    time = np.array([0.0, 1e-9])
    result = AnalysisResult(
        analysis="tran",
        waveforms={
            "time": Waveform("time", np.array([9.0, 10.0]), unit="V", abscissa="time", abscissa_data=time),
            "out": Waveform("out", np.array([0.0, 1.0]), unit="V", abscissa="time", abscissa_data=time),
        },
    )

    with pytest.raises(ValueError, match="abscissa column"):
        result.to_arrays()
    with pytest.raises(ValueError, match="abscissa column"):
        result.to_unit_arrays()


def test_analysis_result_to_arrays_defaults_to_all_waveforms_and_can_copy():
    out = Waveform("out", np.array([0.0, 1.0]))
    result = AnalysisResult("op", {"out": out})

    view_arrays = result.to_arrays()
    copied_arrays = result.to_arrays(copy=True)

    assert list(view_arrays) == ["out"]
    with pytest.raises(ValueError):
        view_arrays["out"][0] = 9.0
    copied_arrays["out"][1] = 8.0
    assert out.data.tolist() == [0.0, 1.0]
    assert copied_arrays["out"].tolist() == [0.0, 8.0]


def test_analysis_result_to_unit_arrays_defaults_to_all_waveforms_and_can_copy():
    out = Waveform("out", np.array([0.0, 1.0]), unit="V")
    result = AnalysisResult("op", {"out": out})

    view_arrays = result.to_unit_arrays()
    copied_arrays = result.to_unit_arrays(copy=True)

    assert list(view_arrays) == ["out"]
    assert view_arrays["out"].unit.symbol == "V"
    with pytest.raises(ValueError):
        view_arrays["out"][0] = 9.0
    copied_arrays["out"][1] = 8.0
    assert out.data.tolist() == [0.0, 1.0]
    assert copied_arrays["out"].values.tolist() == [0.0, 8.0]


def test_analysis_result_to_arrays_can_ignore_missing_waveforms():
    result = AnalysisResult(
        analysis="tran",
        waveforms={"out": Waveform("out", np.array([1.0]))},
    )

    arrays = result.to_arrays("missing", "out", missing="ignore")

    assert list(arrays) == ["out"]
    np.testing.assert_array_equal(arrays["out"], [1.0])


def test_analysis_result_to_unit_arrays_can_ignore_missing_waveforms():
    result = AnalysisResult(
        analysis="tran",
        waveforms={"out": Waveform("out", np.array([1.0]), unit="V")},
    )

    unit_arrays = result.to_unit_arrays("missing", "out", missing="ignore")

    assert list(unit_arrays) == ["out"]
    assert unit_arrays["out"].unit.symbol == "V"
    np.testing.assert_array_equal(unit_arrays["out"].values, [1.0])


def test_analysis_result_to_arrays_rejects_abscissa_column_conflicts():
    time = Waveform("time", np.array([0.0, 1.0]), vector_kind="abscissa")
    result = AnalysisResult(
        analysis="tran",
        waveforms={"time": Waveform("time", np.array([2.0, 3.0]))},
        abscissa=time,
    )

    with pytest.raises(ValueError, match="abscissa column"):
        result.to_arrays()

    arrays = result.to_arrays(include_abscissa=False)
    np.testing.assert_array_equal(arrays["time"], [2.0, 3.0])


def test_analysis_result_to_unit_arrays_rejects_abscissa_column_conflicts():
    time = Waveform("time", np.array([0.0, 1.0]), unit="s", vector_kind="abscissa")
    result = AnalysisResult(
        analysis="tran",
        waveforms={"time": Waveform("time", np.array([2.0, 3.0]), unit="V")},
        abscissa=time,
    )

    with pytest.raises(ValueError, match="abscissa column"):
        result.to_unit_arrays()


def test_analysis_result_to_unit_arrays_requires_unit_metadata():
    result = AnalysisResult(
        analysis="op",
        waveforms={"raw": Waveform("raw", np.array([1.0]))},
    )

    with pytest.raises(ValueError, match="no unit metadata"):
        result.to_unit_arrays()


def test_analysis_result_exposes_named_abscissa_arrays():
    frequency = np.array([1.0, 10.0])
    time = np.array([0.0, 1e-9])
    sweep = np.array([0.0, 1.0])

    ac_result = analysis_result_from_arrays({"out": np.array([1.0, 0.5])}, frequency, {"analysis": "ac"})
    tran_result = analysis_result_from_arrays({"out": np.array([0.0, 1.0])}, time, {"analysis": "tran"})
    dc_result = analysis_result_from_arrays({"out": np.array([0.0, 1.0])}, sweep, {"analysis": "dc"})

    np.testing.assert_allclose(ac_result.abscissa_data, frequency)
    np.testing.assert_allclose(ac_result.frequency, frequency)
    np.testing.assert_allclose(tran_result.time, time)
    np.testing.assert_allclose(dc_result.sweep, sweep)

    with pytest.raises(ValueError, match="not 'time'"):
        ac_result.time
    with pytest.raises(ValueError, match="no abscissa"):
        AnalysisResult("op", {"out": Waveform("out", np.array([1.0]))}).frequency


def test_analysis_result_uses_frequency_abscissa_for_ac_sensitivity():
    frequency = np.array([1.0 + 0.0j, 10.0 + 0.0j])

    result = analysis_result_from_arrays(
        {"v(r1:tc1)": np.array([0.0 + 0.0j, 0.0 + 0.0j])},
        frequency,
        {
            "analysis": "sens",
            "extraction": "rawfile",
            "start": 1.0,
            "stop": 10.0,
            "points": 2,
            "vector_kinds": {"v(r1:tc1)": "sensitivity"},
            "vector_quantities": {"v(r1:tc1)": "sensitivity"},
        },
    )

    assert result.abscissa is not None
    assert result.abscissa.name == "frequency"
    assert result.abscissa.unit == "Hz"
    assert not np.iscomplexobj(result.abscissa.data)
    np.testing.assert_allclose(result.frequency, np.array([1.0, 10.0]))
    waveform = result.waveform("v(r1:tc1)")
    assert waveform.abscissa_name == "frequency"
    assert waveform.vector_kind == "sensitivity"
    np.testing.assert_allclose(waveform.abscissa_data, np.array([1.0, 10.0]))


def test_analysis_result_bode_trace_projects_complex_ac_waveform():
    frequency = np.array([1.0, 10.0, 100.0])
    response = np.array([1.0 + 0.0j, 0.0 - 1.0j, 10.0 + 0.0j])
    result = analysis_result_from_arrays(
        {"out": response},
        frequency,
        {"analysis": "ac"},
        source_vectors={"out": "v(out)"},
    )

    trace = result.bode_trace("v(out)")

    np.testing.assert_allclose(trace.frequency, frequency)
    np.testing.assert_allclose(trace.gain_db, np.array([0.0, 0.0, 20.0]))
    np.testing.assert_allclose(trace.phase, np.array([0.0, -90.0, 0.0]))
    assert trace.phase_unit == "deg"


def test_analysis_result_bode_trace_can_unwrap_phase_in_radians():
    frequency = np.array([1.0, 10.0])
    response = np.exp(1j * np.deg2rad(np.array([170.0, -170.0])))
    result = analysis_result_from_arrays({"out": response}, frequency, {"analysis": "ac"})

    trace = result.bode_trace("out", phase_unit="rad", unwrap_phase=True)

    np.testing.assert_allclose(trace.phase, np.deg2rad(np.array([170.0, 190.0])))
    assert trace.phase_unit == "rad"


def test_analysis_result_bode_trace_requires_frequency_abscissa():
    transient = analysis_result_from_arrays(
        {"out": np.array([1.0 + 0.0j, 0.0 - 1.0j])},
        np.array([0.0, 1e-9]),
        {"analysis": "tran"},
    )
    manual = AnalysisResult(
        "manual",
        {"out": Waveform("out", np.array([1.0 + 0.0j]), abscissa="time", abscissa_data=np.array([0.0]))},
    )

    with pytest.raises(ValueError, match="not 'frequency'"):
        transient.bode_trace("out")
    with pytest.raises(ValueError, match="frequency abscissa"):
        manual.bode_trace("out")


def test_waveform_calculus_uses_explicit_abscissa():
    time = Waveform("time", np.array([0.0, 1.0, 2.0, 3.0]), unit="s", quantity="time")
    waveform = Waveform(
        "out",
        np.array([0.0, 2.0, 4.0, 6.0]),
        unit="V",
        quantity="voltage",
        vector_kind="node_voltage",
    )
    linked = Waveform(
        "out",
        np.array([0.0, 2.0, 4.0, 6.0]),
        unit="V",
        quantity="voltage",
        vector_kind="node_voltage",
        abscissa="time",
        abscissa_data=time.data,
    )

    derivative = waveform.derivative(time)
    integral = waveform.integral(time)
    linked_derivative = linked.derivative()
    linked_integral = linked.integral()

    np.testing.assert_allclose(derivative.data, np.array([2.0, 2.0, 2.0, 2.0]))
    np.testing.assert_allclose(integral.data, np.array([0.0, 1.0, 4.0, 9.0]))
    np.testing.assert_allclose(linked_derivative.data, derivative.data)
    np.testing.assert_allclose(linked_integral.data, integral.data)
    assert derivative.unit == "V/s"
    assert derivative.quantity == "voltage_derivative"
    assert integral.unit == "V*s"
    assert integral.quantity == "voltage_integral"
    assert linked_derivative.unit == "V/s"
    assert linked_integral.unit == "V*s"
    assert derivative.vector_kind == "node_voltage"
    assert integral.vector_kind == "node_voltage"
    assert linked_derivative.vector_kind == "node_voltage"
    assert linked_integral.vector_kind == "node_voltage"
    assert derivative.metadata["derived_from"] == "out"


def test_waveform_calculus_accepts_unit_array_abscissa():
    time = ms(np.array([0.0, 1.0, 2.0, 3.0]))
    waveform = Waveform("out", np.array([0.0, 2.0, 4.0, 6.0]), unit="V", quantity="voltage")

    derivative = waveform.derivative(time)
    integral = waveform.integral(time)

    np.testing.assert_allclose(derivative.data, np.array([2.0, 2.0, 2.0, 2.0]))
    np.testing.assert_allclose(integral.data, np.array([0.0, 1.0, 4.0, 9.0]))
    assert derivative.unit == "V/ms"
    assert integral.unit == "V*ms"


def test_analysis_result_calculus_uses_result_abscissa():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    result = analysis_result_from_arrays(
        {"out": np.array([0.0, 2.0, 4.0, 6.0])},
        time,
        {"analysis": "tran"},
    )

    derivative = result.derivative("out", "slew")
    integral = result.integral("out", "charge")

    assert derivative.name == "slew"
    assert integral.name == "charge"
    np.testing.assert_allclose(derivative.data, np.array([2.0, 2.0, 2.0, 2.0]))
    np.testing.assert_allclose(integral.data, np.array([0.0, 1.0, 4.0, 9.0]))


def test_result_ops_are_explicit_owner_for_analysis_result_transforms():
    result = analysis_result_from_arrays(
        {"out": np.array([0.0, 2.0, 4.0, 6.0])},
        np.array([0.0, 1.0, 2.0, 3.0]),
        {"analysis": "tran"},
    )

    method_derivative = result.derivative("out", "slew")
    op_derivative = result_ops.derivative(result, "out", "slew")
    method_window = result.window("out", 0.5, 2.5, "active")
    op_window = result_ops.window(result, "out", 0.5, 2.5, "active")

    np.testing.assert_allclose(op_derivative.data, method_derivative.data)
    np.testing.assert_allclose(op_window.data, method_window.data)
    assert op_derivative.name == method_derivative.name == "slew"
    assert op_window.name == method_window.name == "active"


def test_analysis_result_calculus_uses_waveform_abscissa_when_result_has_none():
    waveform = Waveform(
        "out",
        np.array([0.0, 2.0, 4.0, 6.0]),
        unit="V",
        quantity="voltage",
        abscissa="time",
        abscissa_data=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    result = AnalysisResult("tran", {"out": waveform})

    derivative = result.derivative("out", "slew")
    integral = result.integral("out", "charge")

    assert derivative.name == "slew"
    assert integral.name == "charge"
    np.testing.assert_allclose(derivative.data, np.array([2.0, 2.0, 2.0, 2.0]))
    np.testing.assert_allclose(integral.data, np.array([0.0, 1.0, 4.0, 9.0]))
    assert derivative.unit == "V/s"
    assert integral.unit == "V*s"


def test_waveform_resample_interpolates_onto_target_abscissa():
    source_time = Waveform("time", np.array([0.0, 1.0, 3.0]), unit="s", quantity="time")
    target_time = Waveform("time", np.array([0.0, 0.5, 2.0, 3.0]), unit="s", quantity="time")
    waveform = Waveform(
        "out",
        np.array([0.0, 2.0, 6.0]),
        unit="V",
        quantity="voltage",
        source_vector="v(out)",
        raw_vector_name="v(out)",
        vector_kind="node_voltage",
        metadata={"corner": "tt"},
    )

    resampled = waveform.resample(target_time, source_abscissa=source_time, name="out_dense")

    assert resampled.name == "out_dense"
    assert resampled.unit == "V"
    assert resampled.quantity == "voltage"
    assert resampled.source_vector == "v(out)"
    assert resampled.raw_vector_name == "v(out)"
    assert resampled.vector_kind == "node_voltage"
    assert resampled.abscissa == "time"
    assert resampled.abscissa_name == "time"
    assert resampled.metadata == {"corner": "tt", "derived_from": "out"}
    np.testing.assert_allclose(resampled.abscissa_data, target_time.data)
    np.testing.assert_allclose(resampled.data, np.array([0.0, 1.0, 4.0, 6.0]))


def test_waveform_resample_accepts_unit_array_abscissas():
    source_time = ms(np.array([0.0, 1.0, 3.0]))
    target_time = ms(np.array([0.0, 0.5, 2.0, 3.0]))
    waveform = Waveform("out", np.array([0.0, 2.0, 6.0]), unit="V", quantity="voltage", abscissa="time")

    resampled = waveform.resample(target_time, source_abscissa=source_time)

    assert resampled.abscissa == "time"
    np.testing.assert_allclose(resampled.abscissa_data, np.array([0.0, 0.5, 2.0, 3.0]))
    np.testing.assert_allclose(resampled.data, np.array([0.0, 1.0, 4.0, 6.0]))


def test_waveform_window_slices_bound_abscissa_and_preserves_metadata():
    waveform = Waveform(
        "out",
        np.array([0.0, 0.5, 1.0, 1.5]),
        unit="V",
        quantity="voltage",
        title="Output",
        source_vector="v(out)",
        raw_vector_name="v(out)",
        vector_kind="node_voltage",
        analysis="tran",
        source="rawfile",
        extraction="rawfile",
        abscissa="time",
        abscissa_data=np.array([0.0, 1.0, 2.0, 3.0]),
        metadata={"corner": "tt"},
    )

    window = waveform.window(0.5, 2.0, name="out_active")

    assert window.name == "out_active"
    assert window.display_name == "out_active"
    assert window.unit == "V"
    assert window.quantity == "voltage"
    assert window.title == "Output"
    assert window.source_vector == "v(out)"
    assert window.raw_vector_name == "v(out)"
    assert window.vector_kind == "node_voltage"
    assert window.analysis == "tran"
    assert window.source == "rawfile"
    assert window.metadata == {"corner": "tt"}
    assert window.abscissa == "time"
    assert window.abscissa_name == "time"
    np.testing.assert_allclose(window.data, np.array([0.5, 1.0]))
    np.testing.assert_allclose(window.abscissa_data, np.array([1.0, 2.0]))


def test_waveform_window_accepts_unit_array_abscissa_and_quantity_bounds():
    source_time = ms(np.array([0.0, 1.0, 2.0, 3.0]))
    waveform = Waveform("out", np.array([0.0, 2.0, 4.0, 6.0]), unit="V", quantity="voltage", abscissa="time")

    window = waveform.window(1 @ ms, 2 @ ms, source_abscissa=source_time)

    assert window.name == "out"
    assert window.abscissa == "time"
    assert window.unit == "V"
    assert window.quantity == "voltage"
    np.testing.assert_allclose(window.data, np.array([2.0, 4.0]))
    np.testing.assert_allclose(window.abscissa_data, np.array([1.0, 2.0]))


def test_analysis_result_window_uses_result_abscissa():
    result = analysis_result_from_arrays(
        {"out": np.array([0.0, 2.0, 4.0, 6.0])},
        np.array([0.0, 1.0, 2.0, 3.0]),
        {"analysis": "tran"},
        source_vectors={"out": "v(out)"},
    )

    window = result.window("out", 0.5, 2.5, "out_active")

    assert window.name == "out_active"
    assert window.abscissa == "time"
    assert window.unit == "V"
    assert window.quantity == "voltage"
    np.testing.assert_allclose(window.data, np.array([2.0, 4.0]))
    np.testing.assert_allclose(window.abscissa_data, np.array([1.0, 2.0]))


def test_analysis_result_windowed_crops_abscissa_and_waveforms():
    result = analysis_result_from_arrays(
        {
            "out": np.array([0.0, 2.0, 4.0, 6.0]),
            "in": np.array([0.0, 1.0, 0.0, -1.0]),
        },
        np.array([0.0, 1.0, 2.0, 3.0]),
        {"analysis": "tran", "run": "smoke"},
        source_vectors={"out": "v(out)", "in": "v(in)"},
    )

    windowed = result.windowed(0.5, 2.5)

    assert windowed.analysis == "tran"
    assert windowed.source == result.source
    assert windowed.metadata == {"analysis": "tran", "run": "smoke"}
    assert windowed.abscissa is not None
    assert windowed.abscissa.name == "time"
    assert windowed.abscissa.unit == "s"
    np.testing.assert_allclose(windowed.abscissa.data, np.array([1.0, 2.0]))
    np.testing.assert_allclose(windowed.time, np.array([1.0, 2.0]))
    np.testing.assert_allclose(windowed["out"].data, np.array([2.0, 4.0]))
    np.testing.assert_allclose(windowed["out"].abscissa_data, np.array([1.0, 2.0]))
    np.testing.assert_allclose(windowed["in"].data, np.array([1.0, 0.0]))
    np.testing.assert_allclose(result["out"].data, np.array([0.0, 2.0, 4.0, 6.0]))


def test_analysis_result_windowed_selects_aliases_and_unit_bounds():
    result = analysis_result_from_arrays(
        {
            "out": np.array([0.0, 2.0, 4.0, 6.0]),
            "in": np.array([0.0, 1.0, 0.0, -1.0]),
        },
        np.array([0.0, 0.001, 0.002, 0.003]),
        {"analysis": "tran"},
        source_vectors={"out": "v(out)", "in": "v(in)"},
    )

    windowed = result.windowed(1 @ ms, 2 @ ms, "v(out)", "missing", missing="ignore")

    assert list(windowed.waveforms) == ["v(out)"]
    assert windowed.abscissa is not None
    np.testing.assert_allclose(windowed.abscissa.data, np.array([0.001, 0.002]))
    np.testing.assert_allclose(windowed["v(out)"].data, np.array([2.0, 4.0]))
    with pytest.raises(WaveformNotFoundError):
        result.windowed(1 @ ms, 2 @ ms, "missing")


def test_analysis_result_windowed_requires_result_abscissa():
    result = AnalysisResult("op", {"out": Waveform("out", np.array([1.0]))})

    with pytest.raises(ValueError, match="no abscissa"):
        result.windowed()


def test_waveform_window_rejects_invalid_ranges_and_bounds():
    waveform = Waveform("out", np.array([1.0, 2.0]), abscissa="time", abscissa_data=np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="less than or equal"):
        waveform.window(2.0, 1.0)
    with pytest.raises(ValueError, match="contains no samples"):
        waveform.window(2.0, 3.0)
    with pytest.raises(ValueError, match="finite abscissa"):
        Waveform("bad", np.array([1.0, 2.0]), abscissa="time", abscissa_data=np.array([0.0, np.nan])).window()
    with pytest.raises(ValueError, match="start bound must be a scalar"):
        waveform.window(np.array([0.0, 1.0]), 1.0)
    with pytest.raises(ValueError, match="abscissa-compatible"):
        waveform.window(1 @ V, 2 @ V, source_abscissa=ms(np.array([0.0, 1.0])))


def test_waveform_sample_at_interpolates_scalar_and_array_targets():
    source_time = Waveform("time", np.array([0.0, 1.0, 3.0]), unit="s", quantity="time")
    waveform = Waveform("out", np.array([0.0, 2.0, 6.0]), unit="V", quantity="voltage")

    scalar = waveform.sample_at(0.5, source_abscissa=source_time)
    samples = waveform.sample_at(ms(np.array([0.0, 0.5, 2.0, 3.0])), source_abscissa=source_time)
    typed_scalar = waveform.sample_at(0.5, source_abscissa=source_time, with_unit=True)
    typed_samples = waveform.sample_at(
        ms(np.array([0.0, 0.5, 2.0, 3.0])),
        source_abscissa=source_time,
        with_unit=True,
    )

    assert scalar == pytest.approx(1.0)
    np.testing.assert_allclose(samples, np.array([0.0, 1.0, 4.0, 6.0]))
    assert isinstance(typed_scalar, Quantity)
    assert typed_scalar.unit is V
    assert typed_scalar.value == pytest.approx(1.0)
    assert isinstance(typed_samples, UnitArray)
    assert typed_samples.unit is V
    np.testing.assert_allclose(typed_samples.values, np.array([0.0, 1.0, 4.0, 6.0]))


def test_analysis_result_sample_at_uses_result_abscissa():
    result = analysis_result_from_arrays(
        {"out": np.array([0.0, 2.0, 6.0])},
        np.array([0.0, 1.0, 3.0]),
        {"analysis": "tran"},
        source_vectors={"out": "v(out)"},
    )

    assert result.sample_at("out", 0.5) == pytest.approx(1.0)
    np.testing.assert_allclose(result.sample_at("out", np.array([0.0, 2.0, 3.0])), np.array([0.0, 4.0, 6.0]))
    typed = result.sample_at("out", 0.5, with_unit=True)
    assert isinstance(typed, Quantity)
    assert typed.unit is V
    assert typed.value == pytest.approx(1.0)


def test_waveform_resample_uses_bound_abscissa_and_preserves_complex_values():
    waveform = Waveform(
        "gain",
        np.array([1.0 + 0.0j, 3.0 + 4.0j]),
        unit="V/V",
        quantity="gain",
        abscissa="frequency",
        abscissa_data=np.array([1.0, 9.0]),
    )

    resampled = waveform.resample(
        np.array([0.0, 1.0, 5.0, 9.0, 10.0]),
        left=-1.0 - 2.0j,
        right=4.0 + 5.0j,
    )

    assert resampled.name == "gain.resampled"
    assert resampled.abscissa == "frequency"
    assert resampled.abscissa_name == "frequency"
    assert np.iscomplexobj(resampled.data)
    np.testing.assert_allclose(resampled.abscissa_data, np.array([0.0, 1.0, 5.0, 9.0, 10.0]))
    np.testing.assert_allclose(
        resampled.data,
        np.array([-1.0 - 2.0j, 1.0 + 0.0j, 2.0 + 2.0j, 3.0 + 4.0j, 4.0 + 5.0j]),
    )


def test_waveform_sample_at_preserves_complex_values_and_boundary_fills():
    waveform = Waveform(
        "gain",
        np.array([1.0 + 0.0j, 3.0 + 4.0j]),
        abscissa="frequency",
        abscissa_data=np.array([1.0, 9.0]),
    )

    samples = waveform.sample_at(np.array([0.0, 5.0, 10.0]), left=-1.0 - 2.0j, right=4.0 + 5.0j)

    assert waveform.sample_at(5.0) == pytest.approx(2.0 + 2.0j)
    assert np.iscomplexobj(samples)
    np.testing.assert_allclose(samples, np.array([-1.0 - 2.0j, 2.0 + 2.0j, 4.0 + 5.0j]))


def test_analysis_result_resample_uses_result_abscissa():
    result = analysis_result_from_arrays(
        {"out": np.array([0.0, 2.0, 6.0])},
        np.array([0.0, 1.0, 3.0]),
        {"analysis": "tran"},
        source_vectors={"out": "v(out)"},
    )

    resampled = result.resample("out", np.array([0.0, 0.5, 2.0, 3.0]), "out_dense")

    assert resampled.name == "out_dense"
    assert resampled.abscissa == "time"
    assert resampled.unit == "V"
    assert resampled.quantity == "voltage"
    np.testing.assert_allclose(resampled.data, np.array([0.0, 1.0, 4.0, 6.0]))


def test_waveform_calculus_requires_matching_abscissa():
    waveform = Waveform("out", np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="no abscissa data"):
        waveform.derivative()
    with pytest.raises(ValueError, match="same length"):
        waveform.derivative(np.array([0.0, 1.0, 2.0]))

    result = AnalysisResult("op", {"out": waveform})
    with pytest.raises(ValueError, match="no abscissa"):
        result.integral("out")


def test_waveform_resample_requires_monotonic_one_dimensional_abscissas():
    waveform = Waveform("out", np.array([1.0, 2.0]), abscissa="time", abscissa_data=np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="strictly increasing"):
        waveform.resample(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="sampling source abscissa must be strictly increasing"):
        waveform.sample_at(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="one-dimensional target"):
        waveform.resample(np.array([[0.0, 1.0]]), source_abscissa=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="scalar or one-dimensional target"):
        waveform.sample_at(np.array([[0.0, 1.0]]), source_abscissa=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="resampling requires finite abscissa values"):
        waveform.resample(np.array([0.0, np.nan]), source_abscissa=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="sampling requires finite abscissa values"):
        waveform.sample_at(np.array([0.0, np.nan]), source_abscissa=np.array([0.0, 1.0]))


def test_analysis_result_from_arrays_sets_analysis_and_extraction_on_waveforms():
    result = analysis_result_from_arrays(
        {"out": np.array([1.0])},
        np.array([1.0]),
        {"analysis": "ac", "extraction": "rawfile"},
        source_vectors={"out": "v(out)"},
    )

    waveform = result.waveform("out")
    assert waveform.analysis == "ac"
    assert waveform.source == "rawfile"
    assert waveform.extraction == "rawfile"


def test_analysis_result_from_arrays_attaches_vector_metadata():
    result = analysis_result_from_arrays(
        {"v(r1:tc1)": np.array([0.0])},
        None,
        {
            "analysis": "sens",
            "extraction": "rawfile",
            "vector_kinds": {"v(r1:tc1)": "sensitivity"},
            "vector_quantities": {"v(r1:tc1)": "sensitivity"},
            "vector_units": {"v(r1:tc1)": None},
            "vector_metadata": {
                "v(r1:tc1)": {
                    "sensitivity_element": "r1",
                    "sensitivity_parameter": "tc1",
                }
            },
        },
    )

    waveform = result.waveform("v(r1:tc1)")

    assert waveform.vector_kind == "sensitivity"
    assert waveform.quantity == "sensitivity"
    assert waveform.unit is None
    assert waveform.metadata == {
        "sensitivity_element": "r1",
        "sensitivity_parameter": "tc1",
    }
