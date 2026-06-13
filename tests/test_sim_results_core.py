from typing import Any, cast

import numpy as np
import pytest

from monata.sim.results import (
    AnalysisResult,
    SimResult,
    Waveform,
    WaveformNotFoundError,
    analysis_result_from_arrays,
)
from monata.sim.vector_names import (
    branch_current_vector,
    device_parameter_vector,
    expression_vector,
    internal_parameter_vector,
    node_current_vector,
    normalize_vector_name,
    voltage_vector,
)
from monata.units import UnitArray, V, ms
pytestmark = pytest.mark.slow

def test_waveform_preserves_typed_metadata():
    data = np.array([0.0, 1.0])
    waveform = Waveform(
        name="out",
        data=data,
        unit="V",
        quantity="voltage",
        title="Output",
        source_vector="v(out)",
        abscissa="time",
    )

    assert waveform.name == "out"
    np.testing.assert_allclose(waveform.data, data)
    assert waveform.unit == "V"
    assert waveform.quantity == "voltage"
    assert waveform.title == "Output"
    assert waveform.source_vector == "v(out)"
    assert waveform.raw_vector_name == "v(out)"
    assert waveform.display_name == "out"
    assert waveform.normalized_name == "out"
    assert waveform.abscissa == "time"


def test_waveform_public_payloads_are_recursively_read_only():
    source_data = np.array([0.0, 1.0])
    source_metadata = {
        "limits": np.array([0.0, 1.0]),
        "nested": {"labels": ["low", "high"]},
    }
    waveform = Waveform("out", source_data, metadata=source_metadata)

    source_data[0] = 9.0
    cast(Any, source_metadata["limits"])[0] = 9.0
    cast(Any, source_metadata["nested"])["labels"].append("extra")

    np.testing.assert_allclose(waveform.data, [0.0, 1.0])
    np.testing.assert_allclose(waveform.metadata["limits"], [0.0, 1.0])
    assert waveform.metadata["nested"]["labels"] == ["low", "high"]
    with pytest.raises(ValueError):
        waveform.data[0] = 9.0
    with pytest.raises(TypeError):
        cast(Any, waveform.metadata)["new"] = "value"
    with pytest.raises(ValueError):
        cast(Any, waveform.metadata["limits"])[0] = 9.0
    with pytest.raises(TypeError):
        cast(Any, waveform.metadata["nested"])["labels"].append("extra")


def test_waveform_can_project_to_unit_array_and_convert_units():
    waveform = Waveform(
        name="out",
        data=np.array([0.0, 1.2]),
        unit="V",
        quantity="voltage",
        vector_kind="node_voltage",
    )

    unit_array = waveform.unit_array()
    converted = waveform.to_unit("mV", name="out_mv")

    assert isinstance(unit_array, UnitArray)
    assert unit_array.unit.symbol == "V"
    assert converted.name == "out_mv"
    assert converted.unit == "mV"
    assert converted.quantity == "voltage"
    assert converted.vector_kind == "node_voltage"
    np.testing.assert_allclose(converted.data, np.array([0.0, 1200.0]))


def test_waveform_from_array_accepts_abscissa_data():
    waveform = Waveform.from_array(
        "out",
        np.array([0.0, 1.2]),
        title="Output voltage",
        abscissa=np.array([0.0, 1e-9]),
        abscissa_name="time",
        unit="V",
        quantity="voltage",
        source="manual",
    )

    assert waveform.name == "out"
    assert waveform.title == "Output voltage"
    assert waveform.abscissa_name == "time"
    assert waveform.unit == "V"
    assert waveform.quantity == "voltage"
    assert waveform.metadata == {"source": "manual"}
    np.testing.assert_allclose(waveform.data, np.array([0.0, 1.2]))
    np.testing.assert_allclose(waveform.abscissa_data, np.array([0.0, 1e-9]))


def test_waveform_from_unit_array_uses_unit_symbol_and_abscissa_waveform():
    time = Waveform.from_unit_array("time", ms(np.array([0.0, 1.0])), quantity="time")
    waveform = Waveform.from_unit_array(
        "out",
        V(np.array([0.0, 1.2])),
        title="Output voltage",
        abscissa=time,
        quantity="voltage",
        source="unit_array",
    )

    assert waveform.unit == "V"
    assert waveform.abscissa_name == "time"
    assert waveform.metadata == {"source": "unit_array"}
    np.testing.assert_allclose(waveform.data, np.array([0.0, 1.2]))
    np.testing.assert_allclose(waveform.abscissa_data, time.data)
    with pytest.raises(TypeError, match="UnitArray"):
        Waveform.from_unit_array("bad", np.array([1.0]))  # type: ignore[arg-type]


def test_waveform_uses_monata_unit_array_constructor_name_only():
    assert not hasattr(Waveform, "from_unit_values")


def test_waveform_unit_array_requires_unit_metadata():
    with pytest.raises(ValueError, match="no unit metadata"):
        Waveform(name="raw", data=np.array([1.0])).unit_array()


def test_waveform_string_display_and_data_repr():
    waveform = Waveform(name="out", data=np.array([0.0, 1.2]), title="Output voltage")
    untitled = Waveform(name="in", data=np.array([1.0]))

    assert str(waveform) == "Output voltage"
    assert str(untitled) == "in"
    assert waveform.str_data() == "array([0. , 1.2])"
    assert repr(waveform) == "Waveform out array([0. , 1.2])"


def test_waveform_clone_and_with_title_update_metadata_immutably():
    waveform = Waveform(
        name="out",
        data=np.array([0.0, 1.2]),
        title="Output voltage",
        unit="V",
        abscissa="time",
        abscissa_data=np.array([0.0, 1e-9]),
        metadata={"corner": "tt"},
    )

    retitled = waveform.with_title("Settled output")
    cleared = retitled.with_title(None)
    cloned = waveform.clone(name="out_alt", data=np.array([0.1, 1.1]), title=123)

    assert waveform.title == "Output voltage"
    assert waveform.plot_name == "Output voltage"
    assert retitled.title == "Settled output"
    assert retitled.plot_name == "Settled output"
    assert str(retitled) == "Settled output"
    assert cleared.title is None
    assert cleared.plot_name is None
    assert str(cleared) == "out"
    assert cloned.name == "out_alt"
    assert cloned.title == "123"
    assert cloned.plot_name == "123"
    assert cloned.unit == "V"
    assert cloned.metadata == {"corner": "tt"}
    np.testing.assert_allclose(cloned.data, np.array([0.1, 1.1]))
    np.testing.assert_allclose(cloned.abscissa_data, np.array([0.0, 1e-9]))
    np.testing.assert_allclose(waveform.data, np.array([0.0, 1.2]))


def test_waveform_abscissa_data_is_validated():
    waveform = Waveform(name="out", data=np.array([0.0, 1.0]), abscissa_data=np.array([10.0, 20.0]))

    np.testing.assert_allclose(waveform.abscissa_data, np.array([10.0, 20.0]))
    with pytest.raises(ValueError, match="same length"):
        Waveform(name="bad", data=np.array([0.0, 1.0]), abscissa_data=np.array([1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        Waveform(name="bad", data=np.array([0.0, 1.0]), abscissa_data=np.array([[1.0, 2.0]]))


def test_sim_result_array_constructor_synthesizes_analysis_result():
    sweep = np.array([0.0, 0.5, 1.0])
    result = SimResult(
        status="ok",
        waveforms={"in": sweep},
        sweep_var=sweep,
        corner=None,
        metadata={"analysis": "dc", "extraction": "wrdata"},
    )

    assert result.waveforms["in"] is not sweep
    assert result.sweep_var is not sweep
    assert not result.waveforms["in"].flags.writeable
    assert result.sweep_var is not None
    assert not result.sweep_var.flags.writeable
    assert result.analysis_result is not None
    assert result.analysis_result.analysis == "dc"
    assert result.analysis_result.source == "wrdata"
    np.testing.assert_allclose(result.analysis_result.waveform("in").data, sweep)
    with pytest.raises(ValueError):
        result.waveforms["in"][0] = 9.0
    with pytest.raises(ValueError):
        result.sweep_var[0] = 9.0


def test_sim_result_preserves_explicit_analysis_result():
    typed = AnalysisResult(
        analysis="op",
        waveforms={"in": Waveform("in", np.array([1.2]), unit="V")},
        source="stdout-print",
    )
    result = SimResult(
        status="ok",
        waveforms={"in": np.array([1.2])},
        sweep_var=None,
        corner=None,
        analysis_result=typed,
    )

    assert result.analysis_result is typed
    assert "in" in result.waveforms


def test_failed_sim_result_has_no_success_analysis_result():
    result = SimResult(
        status="failed",
        waveforms={},
        sweep_var=None,
        corner=None,
        error_message="failed",
    )

    assert result.analysis_result is None


def test_rawfile_and_fallback_arrays_build_same_typed_shape():
    sweep = np.array([1.0, 10.0])
    values = np.array([0.1, 0.2])

    rawfile = analysis_result_from_arrays(
        {"out": values},
        sweep,
        {"analysis": "ac", "extraction": "rawfile"},
        source_vectors={"out": "mag(v(out))"},
    )
    fallback = analysis_result_from_arrays(
        {"out": values},
        sweep,
        {"analysis": "ac", "extraction": "wrdata"},
        source_vectors={"out": "mag(v(out))"},
    )

    assert set(rawfile.waveforms) == set(fallback.waveforms)
    assert rawfile.abscissa is not None
    assert fallback.abscissa is not None
    assert rawfile.abscissa.name == fallback.abscissa.name == "frequency"
    np.testing.assert_allclose(rawfile.waveform("out").data, fallback.waveform("out").data)


def test_non_voltage_waveform_names_are_not_mislabeled_as_voltage():
    result = analysis_result_from_arrays(
        {"gain": np.array([1.0, 2.0]), "i(v1)": np.array([0.1, 0.2])},
        np.array([1.0, 10.0]),
        {"analysis": "ac", "extraction": "rawfile"},
    )

    assert result.waveform("gain").quantity is None
    assert result.waveform("gain").unit is None
    assert result.waveform("i(v1)").quantity == "current"
    assert result.waveform("i(v1)").unit == "A"
    assert result.waveform("i(v1)").vector_kind == "branch_current"


def test_sim_result_waveform_helpers_delegate_to_typed_result_lookup():
    result = SimResult(
        status="ok",
        waveforms={"out": np.array([1.0])},
        sweep_var=np.array([0.0]),
        corner=None,
        metadata={
            "analysis": "tran",
            "vector_raw_names": {"out": "v(out)"},
        },
    )

    selected = result.select_waveforms("v(out)", "out")

    assert result.waveform("v(out)").name == "out"
    assert list(selected) == ["v(out)", "out"]
    assert selected["v(out)"].name == "out"
    assert selected["out"].name == "out"


def test_failed_sim_result_waveform_helpers_report_missing_success_result():
    result = SimResult(
        status="failed",
        waveforms={},
        sweep_var=None,
        corner=None,
        error_message="failed",
    )

    with pytest.raises(WaveformNotFoundError, match="no successful analysis result"):
        result.waveform("out")
    assert result.select_waveforms("out", missing="ignore") == {}


def test_missing_waveform_error_includes_lookup_context():
    result = AnalysisResult(
        analysis="tran",
        waveforms={"out": Waveform("out", np.array([1.0]))},
    )

    with pytest.raises(WaveformNotFoundError) as exc_info:
        result.waveform("missing(node)")

    message = str(exc_info.value)
    assert "missing(node)" in message
    assert "missing_node" in message
    assert "out" in message


def test_result_accessors_group_waveforms_by_quantity_and_kind():
    result = AnalysisResult(
        analysis="ac",
        waveforms={
            "out": Waveform("out", np.array([1.0]), quantity="voltage", vector_kind="node_voltage"),
            "v(out,in)": Waveform("v(out,in)", np.array([0.5]), quantity="voltage", vector_kind="differential_voltage"),
            "i(v1)": Waveform("i(v1)", np.array([0.1]), quantity="current", vector_kind="branch_current"),
            "@m1[id]": Waveform("@m1[id]", np.array([0.2]), quantity="current", vector_kind="node_current"),
            "@m1[gm]": Waveform("@m1[gm]", np.array([1e-3]), vector_kind="element_parameter"),
            "@temp": Waveform("@temp", np.array([27.0]), vector_kind="internal_parameter"),
            "onoise_spectrum": Waveform(
                "onoise_spectrum",
                np.array([1e-9]),
                quantity="noise",
                vector_kind="noise_spectrum",
            ),
            "onoise_total": Waveform(
                "onoise_total",
                np.array([2e-9]),
                quantity="noise",
                vector_kind="noise_total",
            ),
        },
    )

    assert set(result.voltages) == {"out", "v(out,in)"}
    assert set(result.node_voltages) == {"out"}
    assert set(result.differential_voltages) == {"v(out,in)"}
    assert set(result.currents) == {"i(v1)", "@m1[id]"}
    assert set(result.branch_currents) == {"i(v1)"}
    assert set(result.node_currents) == {"@m1[id]"}
    assert set(result.element_parameters) == {"@m1[gm]"}
    assert not hasattr(result, "elements")
    assert set(result.device_parameters) == {"@m1[id]", "@m1[gm]"}
    assert set(result.internal_parameters) == {"@temp"}
    assert set(result.noise) == {"onoise_spectrum", "onoise_total"}
    assert set(result.noise_spectra) == {"onoise_spectrum"}
    assert set(result.noise_totals) == {"onoise_total"}


def test_result_accessors_group_advanced_analysis_waveforms_by_kind():
    result = AnalysisResult(
        analysis="mixed",
        waveforms={
            "sens": Waveform("sens", np.array([1.0]), vector_kind="sensitivity"),
            "pole1": Waveform("pole1", np.array([-1.0 + 2.0j]), vector_kind="pole"),
            "zero1": Waveform("zero1", np.array([-3.0 + 0.0j]), vector_kind="zero"),
            "hd2": Waveform("hd2", np.array([1e-3]), vector_kind="distortion"),
            "gain": Waveform("gain", np.array([10.0]), vector_kind="transfer_function"),
            "fourier1": Waveform("fourier1", np.array([0.5]), vector_kind="fourier_component"),
            "db(v(out))": Waveform("db(v(out))", np.array([6.0]), vector_kind="expression"),
            "mag(out)": Waveform("mag(out)", np.array([1.0]), vector_kind="ac_component"),
        },
    )

    assert set(result.sensitivities) == {"sens"}
    assert set(result.poles) == {"pole1"}
    assert set(result.zeros) == {"zero1"}
    assert set(result.pole_zero) == {"pole1", "zero1"}
    assert set(result.distortion) == {"hd2"}
    assert set(result.transfer_functions) == {"gain"}
    assert set(result.fourier_components) == {"fourier1"}
    assert set(result.expressions) == {"db(v(out))"}
    assert set(result.ac_components) == {"mag(out)"}


def test_result_device_parameters_are_grouped_by_element_and_parameter():
    gm = Waveform(
        "@M1[gm]",
        np.array([1e-3]),
        source_vector="@M1[gm]",
        raw_vector_name="@m1[gm]",
        vector_kind="element_parameter",
    )
    drain_current = Waveform(
        "@M1[id]",
        np.array([0.2]),
        source_vector="@M1[id]",
        raw_vector_name="@m1[id]",
        vector_kind="node_current",
        quantity="current",
    )
    resistor_tempco = Waveform(
        "@Rload[tc1]",
        np.array([1e-6]),
        source_vector="@Rload[tc1]",
        raw_vector_name="@rload[tc1]",
        vector_kind="internal_parameter",
    )
    global_temperature = Waveform(
        "@temp",
        np.array([27.0]),
        vector_kind="internal_parameter",
    )
    result = AnalysisResult(
        analysis="op",
        waveforms={
            "@M1[gm]": gm,
            "@M1[id]": drain_current,
            "@Rload[tc1]": resistor_tempco,
            "@temp": global_temperature,
        },
    )

    assert set(result.device_parameters) == {"@M1[gm]", "@M1[id]", "@Rload[tc1]"}
    assert set(result.device_parameters_by_element) == {"M1", "Rload"}
    assert set(result.device_parameters_by_element["M1"]) == {"gm", "id"}
    assert result.device_parameters_by_element["M1"]["gm"] is gm
    assert result.device_parameters_by_element["M1"]["id"] is drain_current
    assert result.device_parameters_by_element["Rload"]["tc1"] is resistor_tempco
    assert "@temp" not in result.device_parameters


def test_result_entity_keyed_accessors_hide_backend_vector_syntax():
    node = Waveform(
        "Out",
        np.array([1.0]),
        quantity="voltage",
        source_vector="v(Out)",
        raw_vector_name="v(out)",
        vector_kind="node_voltage",
    )
    branch = Waveform(
        "i(Vinput)",
        np.array([0.1]),
        quantity="current",
        source_vector="i(Vinput)",
        raw_vector_name="i(vinput)",
        vector_kind="branch_current",
    )
    bare_node = Waveform(
        "fallback",
        np.array([0.5]),
        quantity="voltage",
        vector_kind="node_voltage",
    )
    result = AnalysisResult(
        analysis="tran",
        waveforms={
            "v(Out)": node,
            "i(Vinput)": branch,
            "fallback": bare_node,
        },
    )

    assert set(result.node_voltages) == {"v(Out)", "fallback"}
    assert set(result.branch_currents) == {"i(Vinput)"}
    assert set(result.node_voltages_by_node) == {"Out", "fallback"}
    assert not hasattr(result, "nodes")
    assert result.node_voltages_by_node["Out"] is node
    assert result.node_voltages_by_node["fallback"] is bare_node
    assert set(result.branch_currents_by_element) == {"Vinput"}
    assert not hasattr(result, "branches")
    assert result.branch_currents_by_element["Vinput"] is branch


def test_waveform_is_array_like_without_losing_metadata():
    waveform = Waveform(
        name="out",
        data=np.array([0.0, 0.5, 1.0]),
        unit="V",
        quantity="voltage",
        title="Output",
        source_vector="v(out)",
    )

    assert len(waveform) == 3
    assert waveform.shape == (3,)
    assert waveform.size == 3
    assert waveform.dtype == np.dtype(float)
    assert waveform[1] == pytest.approx(0.5)
    assert list(waveform) == [0.0, 0.5, 1.0]
    np.testing.assert_allclose(np.asarray(waveform), np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(np.asarray(waveform, dtype=np.float32), np.array([0.0, 0.5, 1.0], dtype=np.float32))
    np.testing.assert_allclose(np.sin(waveform), np.sin(waveform.data))

    view = waveform.as_array()
    with pytest.raises(ValueError):
        view[0] = 9.0
    assert waveform.data[0] == pytest.approx(0.0)
    second_view = waveform.as_array()
    with pytest.raises(ValueError):
        second_view[2] = 8.0
    assert waveform.data[2] == pytest.approx(1.0)

    copied = waveform.as_array(copy=True)
    copied[1] = 7.0
    assert waveform.data[1] == pytest.approx(0.5)
    scaled = Waveform("millivolts", np.array([0.0, 1200.0]), unit="mV").to_unit("V").as_array()
    np.testing.assert_allclose(scaled, np.array([0.0, 1.2]))
    assert waveform.unit == "V"
    assert waveform.raw_vector_name == "v(out)"
    assert not hasattr(waveform, "as_ndarray")


def test_waveform_slices_preserve_metadata_and_slice_abscissa():
    waveform = Waveform(
        name="out",
        data=np.array([0.0, 0.5, 1.0, 1.5]),
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
        abscissa_data=np.array([0.0, 1e-9, 2e-9, 3e-9]),
        metadata={"corner": "tt"},
    )

    window = waveform[1:3]
    masked = waveform[np.array([True, False, True, False])]

    assert waveform[2] == pytest.approx(1.0)
    assert isinstance(window, Waveform)
    window_waveform = cast(Waveform, window)
    assert window_waveform.name == "out"
    assert window_waveform.unit == "V"
    assert window_waveform.quantity == "voltage"
    assert window_waveform.title == "Output"
    assert window_waveform.source_vector == "v(out)"
    assert window_waveform.raw_vector_name == "v(out)"
    assert window_waveform.vector_kind == "node_voltage"
    assert window_waveform.analysis == "tran"
    assert window_waveform.source == "rawfile"
    assert window_waveform.metadata == {"corner": "tt"}
    np.testing.assert_allclose(window_waveform.data, np.array([0.5, 1.0]))
    np.testing.assert_allclose(window_waveform.abscissa_data, np.array([1e-9, 2e-9]))
    assert isinstance(masked, Waveform)
    masked_waveform = cast(Waveform, masked)
    np.testing.assert_allclose(masked_waveform.data, np.array([0.0, 1.0]))
    np.testing.assert_allclose(masked_waveform.abscissa_data, np.array([0.0, 2e-9]))


def test_waveform_numpy_ufuncs_preserve_result_metadata_when_units_remain_valid():
    time = np.array([0.0, 1.0])
    waveform = Waveform(
        name="out",
        data=np.array([1.0, 2.0]),
        unit="V",
        quantity="voltage",
        title="Output",
        source_vector="v(out)",
        abscissa="time",
        abscissa_data=time,
        vector_kind="node_voltage",
        metadata={"corner": "tt"},
    )

    squared = np.square(waveform)
    shifted = np.add(waveform, 1.0)
    comparison = np.greater(waveform, 1.5)
    sine = np.sin(waveform)

    assert isinstance(squared, Waveform)
    squared_waveform = cast(Waveform, squared)
    assert squared_waveform.name == "square(out)"
    assert squared_waveform.unit == "V^2"
    assert squared_waveform.quantity == "voltage_square"
    assert squared_waveform.source_vector == "v(out)"
    assert squared_waveform.raw_vector_name == "v(out)"
    assert squared_waveform.vector_kind == "node_voltage"
    assert squared_waveform.abscissa_name == "time"
    assert squared_waveform.metadata == {"corner": "tt", "derived_from": "out"}
    np.testing.assert_allclose(squared_waveform.data, np.array([1.0, 4.0]))
    np.testing.assert_allclose(squared_waveform.abscissa_data, time)
    assert isinstance(shifted, Waveform)
    shifted_waveform = cast(Waveform, shifted)
    assert shifted_waveform.unit == "V"
    assert shifted_waveform.quantity == "voltage"
    assert shifted_waveform.vector_kind == "node_voltage"
    np.testing.assert_allclose(shifted_waveform.data, np.array([2.0, 3.0]))
    assert isinstance(comparison, np.ndarray)
    np.testing.assert_array_equal(comparison, np.array([False, True]))
    assert isinstance(sine, np.ndarray)
    np.testing.assert_allclose(sine, np.sin(waveform.data))


@pytest.mark.parametrize(
    ("raw", "kind", "quantity", "normalized"),
    [
        ("v(out)", "node_voltage", "voltage", "out"),
        ("v(in,out)", "differential_voltage", "voltage", "in_out"),
        ("i(v1)", "branch_current", "current", "i_v1"),
        ("v1#branch", "branch_current", "current", "i_v1"),
        ("@m1[id]", "element_parameter", None, "m1_id"),
        ("@temp", "internal_parameter", None, "temp"),
        ("db(v(out))", "expression", None, "db_v_out"),
        ("v(out)-v(in)", "expression", None, "v_out_v_in"),
    ],
)
def test_normalize_vector_name_classifies_common_ngspice_vectors(raw, kind, quantity, normalized):
    vector = normalize_vector_name(raw)

    assert vector.raw_vector_name == raw
    assert vector.vector_kind == kind
    assert vector.quantity == quantity
    assert vector.normalized_name == normalized


def test_vector_name_helpers_build_classified_spice_probe_names():
    voltage = voltage_vector("out")
    differential = voltage_vector("out", "in")
    current = branch_current_vector("V1")
    node_current = node_current_vector("M1", "id")
    device_parameter = device_parameter_vector("M1", "gm")
    internal_parameter = internal_parameter_vector("temp")
    expression = expression_vector("v(out)-v(in)")

    assert voltage == "v(out)"
    assert differential == "v(out,in)"
    assert current == "i(V1)"
    assert node_current == "@M1[id]"
    assert device_parameter == "@M1[gm]"
    assert internal_parameter == "@temp"
    assert expression == "v(out)-v(in)"
    assert normalize_vector_name(voltage).vector_kind == "node_voltage"
    assert normalize_vector_name(differential).vector_kind == "differential_voltage"
    assert normalize_vector_name(current).vector_kind == "branch_current"
    assert normalize_vector_name(node_current).vector_kind == "element_parameter"
    assert normalize_vector_name(device_parameter).vector_kind == "element_parameter"
    assert normalize_vector_name(internal_parameter).vector_kind == "internal_parameter"
    assert normalize_vector_name(expression).vector_kind == "expression"


def test_vector_name_helpers_reject_empty_ambiguous_or_unsafe_parts():
    with pytest.raises(ValueError, match="voltage node is required"):
        voltage_vector("")
    with pytest.raises(ValueError, match="control characters"):
        voltage_vector("out\nquit")
    with pytest.raises(ValueError, match="whitespace"):
        branch_current_vector("V 1")
    with pytest.raises(ValueError, match="vector delimiters"):
        voltage_vector("out,in")
    with pytest.raises(ValueError, match="device parameter delimiters"):
        device_parameter_vector("M1[bad]", "gm")
    with pytest.raises(ValueError, match="whitespace"):
        expression_vector("v(out) - v(in)")


def test_internal_parameter_is_not_grouped_as_current_by_default():
    result = analysis_result_from_arrays({"@m1[id]": np.array([1.0])}, None, {"analysis": "op"})

    assert result.waveform("@m1[id]").vector_kind == "element_parameter"
    assert result.waveform("@m1[id]").quantity is None
    assert result.currents == {}
