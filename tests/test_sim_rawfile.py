import numpy as np
import pytest

from monata.sim.rawfile import load_ngspice_rawfile


def test_load_ngspice_rawfile_returns_typed_analysis_result(tmp_path):
    rawfile = tmp_path / "dc.raw"
    rawfile.write_text(
        """Title: dc run
Date: today
Plotname: DC transfer characteristic
Flags: real
No. Variables: 2
No. Points: 3
Variables:
    0   v-sweep voltage
    1   v(in)   voltage
Values:
    0   0.0
        0.0
    1   0.5
        0.5
    2   1.0
        1.0
"""
    )

    result = load_ngspice_rawfile(rawfile)

    assert result.analysis == "dc"
    assert result.source == "rawfile"
    assert result.metadata["simulator"] == "ngspice"
    assert result.metadata["source_path"] == str(rawfile)
    assert result.metadata["scale_vector"] == "v-sweep"
    assert result.abscissa is not None
    assert result.abscissa.name == "v-sweep"
    assert result.abscissa.unit == "V"
    assert result.abscissa.vector_kind == "abscissa"
    assert set(result.waveforms) == {"v(in)"}
    assert result.waveform("in").unit == "V"
    assert result.waveform("in").quantity == "voltage"
    assert result.waveform("in").vector_kind == "node_voltage"
    assert result.waveform("in").abscissa == "v-sweep"
    np.testing.assert_allclose(result.abscissa.data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(result.waveform("v(in)").data, np.array([0.0, 0.5, 1.0]))


def test_load_ngspice_rawfile_preserves_complex_ac_vectors(tmp_path):
    rawfile = tmp_path / "ac.raw"
    rawfile.write_text(
        """Title: ac run
Plotname: AC Analysis
Flags: complex
No. Variables: 2
No. Points: 2
Variables:
    0   frequency frequency
    1   v(out) voltage
Values:
    0   1.0,0.0
        2.0,3.0
    1   10.0,0.0
        4.0,5.0
"""
    )

    result = load_ngspice_rawfile(rawfile)

    assert result.analysis == "ac"
    assert result.abscissa is not None
    assert result.abscissa.name == "frequency"
    assert result.abscissa.unit == "Hz"
    assert not np.iscomplexobj(result.abscissa.data)
    np.testing.assert_allclose(result.frequency, np.array([1.0, 10.0]))
    np.testing.assert_allclose(result.waveform("out").abscissa_data, np.array([1.0, 10.0]))
    assert np.iscomplexobj(result.waveform("out").data)
    np.testing.assert_allclose(result.waveform("out").magnitude().data, np.array([np.hypot(2.0, 3.0), np.hypot(4.0, 5.0)]))


def test_load_ngspice_operating_point_rawfile_keeps_first_variable(tmp_path):
    rawfile = tmp_path / "op.raw"
    rawfile.write_text(
        """Title: op run
Plotname: Operating Point
Flags: real
No. Variables: 2
No. Points: 1
Variables:
    0   v(out) voltage
    1   i(v1)  current
Values:
    0   1.2
        -0.001
"""
    )

    result = load_ngspice_rawfile(rawfile, element_names=["V1"])

    assert result.analysis == "op"
    assert result.abscissa is None
    assert "scale_vector" not in result.metadata
    assert set(result.waveforms) == {"v(out)", "i(V1)"}
    assert result.waveform("out").abscissa is None
    assert result.waveform("out").unit == "V"
    assert result.waveform("i(V1)").vector_kind == "branch_current"
    np.testing.assert_allclose(result.waveform("out").data, np.array([1.2]))
    np.testing.assert_allclose(result.waveform("i(V1)").data, np.array([-0.001]))


def test_load_ngspice_rawfile_filters_by_normalized_output_names(tmp_path):
    rawfile = tmp_path / "tran.raw"
    rawfile.write_text(
        """Title: tran run
Plotname: Transient Analysis
Flags: real
No. Variables: 3
No. Points: 2
Variables:
    0   time time
    1   v(in) voltage
    2   v(out) voltage
Values:
    0   0.0
        0.0
        1.0
    1   1e-9
        0.5
        0.2
"""
    )

    result = load_ngspice_rawfile(rawfile, output_names=["out", "v(out)"])

    assert result.analysis == "tran"
    assert set(result.waveforms) == {"v(out)"}
    assert result.waveform("out").normalized_name == "out"
    with pytest.raises(KeyError, match="not found"):
        result.waveform("in")


def test_load_ngspice_rawfile_can_restore_vector_case_for_display(tmp_path):
    rawfile = tmp_path / "tran.raw"
    rawfile.write_text(
        """Title: tran run
Plotname: Transient Analysis
Flags: real
No. Variables: 5
No. Points: 2
Variables:
    0   time        time
    1   v(out)      voltage
    2   v(out,ref)  voltage
    3   i(vinput)   current
    4   vinput#branch current
Values:
    0   0.0
        1.0
        0.5
        0.01
        0.02
    1   1e-9
        2.0
        1.5
        0.02
        0.03
"""
    )

    result = load_ngspice_rawfile(
        rawfile,
        node_names=["Out", "Ref"],
        element_names=["Vinput"],
        output_names=["out", "v(Out,Ref)", "i(Vinput)", "Vinput#branch"],
    )

    assert set(result.waveforms) == {"v(Out)", "v(Out,Ref)", "i(Vinput)", "Vinput#branch"}
    assert result.metadata["rawfile_case_map"] == {
        "nodes": {"out": "Out", "ref": "Ref"},
        "elements": {"vinput": "Vinput"},
    }
    assert result.metadata["vector_raw_names"]["v(Out)"] == "v(out)"

    out = result.waveform("v(out)")
    assert result.waveform("v(Out)") is out
    assert out.name == "Out"
    assert out.source_vector == "v(Out)"
    assert out.raw_vector_name == "v(out)"
    assert out.normalized_name == "out"

    diff = result.waveform("v(out,ref)")
    assert diff.name == "Out,Ref"
    assert diff.source_vector == "v(Out,Ref)"
    assert diff.vector_kind == "differential_voltage"
    assert result.waveform("i(vinput)").source_vector == "i(Vinput)"
    branch_alias = result.waveform("Vinput#branch")
    assert branch_alias.source_vector == "Vinput#branch"
    assert branch_alias.raw_vector_name == "vinput#branch"
    assert branch_alias.vector_kind == "branch_current"
    assert result.branch_currents_by_element["Vinput"] is branch_alias


def test_load_ngspice_rawfile_classifies_saved_device_currents_separately(tmp_path):
    rawfile = tmp_path / "tran.raw"
    rawfile.write_text(
        """Title: saved currents
Plotname: Transient Analysis
Flags: real
No. Variables: 4
No. Points: 2
Variables:
    0   time time
    1   @m1[id] current
    2   @m1[gm] conductance
    3   i(vdd) current
Values:
    0   0.0
        0.001
        0.002
        -0.001
    1   1e-9
        0.0015
        0.0025
        -0.0015
"""
    )

    result = load_ngspice_rawfile(rawfile, element_names=["VDD"])

    drain = result.waveform("@m1[id]")
    gm = result.waveform("@m1[gm]")
    branch = result.waveform("i(VDD)")

    assert drain.vector_kind == "node_current"
    assert drain.quantity == "current"
    assert gm.vector_kind == "element_parameter"
    assert branch.vector_kind == "branch_current"
    assert set(result.node_currents) == {"@m1[id]"}
    assert set(result.branch_currents) == {"i(VDD)"}
    assert result.internal_parameters == {}
    assert set(result.element_parameters) == {"@m1[gm]"}
    assert set(result.currents) == {"@m1[id]", "i(VDD)"}
    assert set(result.device_parameters) == {"@m1[id]", "@m1[gm]"}
    np.testing.assert_allclose(drain.data, np.array([0.001, 0.0015]))
    np.testing.assert_allclose(gm.data, np.array([0.002, 0.0025]))
    np.testing.assert_allclose(branch.data, np.array([-0.001, -0.0015]))


def test_load_ngspice_rawfile_rejects_scale_vector_as_output(tmp_path):
    rawfile = tmp_path / "tran.raw"
    rawfile.write_text(
        """Title: tran run
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 1
Variables:
    0   time time
    1   v(out) voltage
Values:
    0   0.0
        1.0
"""
    )

    with pytest.raises(KeyError, match="scale vector"):
        load_ngspice_rawfile(rawfile, output_names=["time"])


def test_load_ngspice_rawfile_can_override_analysis_name(tmp_path):
    rawfile = tmp_path / "unknown.raw"
    rawfile.write_text(
        """Title: custom
Plotname: custom plot
Flags: real
No. Variables: 2
No. Points: 1
Variables:
    0   frequency frequency
    1   v(out) voltage
Values:
    0   1.0
        2.0
"""
    )

    result = load_ngspice_rawfile(rawfile, analysis="ac")

    assert result.analysis == "ac"
    assert result.metadata["analysis"] == "ac"


@pytest.mark.parametrize(
    ("plotname", "analysis"),
    [
        ("Sensitivity Analysis", "sens"),
        ("AC Sensitivity Analysis", "sens"),
        ("Pole-Zero Analysis", "pz"),
        ("Pole Zero Analysis", "pz"),
        ("Distortion Analysis", "disto"),
        ("Transfer Function Analysis", "tf"),
        ("Fourier Analysis", "four"),
        ("DC transfer characteristic", "dc"),
    ],
)
def test_load_ngspice_rawfile_infers_extended_analysis_names(tmp_path, plotname, analysis):
    rawfile = tmp_path / f"{analysis}.raw"
    rawfile.write_text(
        f"""Title: analysis inference
Plotname: {plotname}
Flags: real
No. Variables: 2
No. Points: 1
Variables:
    0   frequency frequency
    1   v(out) voltage
Values:
    0   1.0
        2.0
"""
    )

    result = load_ngspice_rawfile(rawfile)

    assert result.analysis == analysis
    assert result.metadata["analysis"] == analysis


def test_load_ngspice_rawfile_types_sensitivity_vectors(tmp_path):
    rawfile = tmp_path / "sens.raw"
    rawfile.write_text(
        """Title: sensitivity
Plotname: AC Sensitivity Analysis
Flags: real
No. Variables: 2
No. Points: 2
Variables:
    0   frequency frequency
    1   v(r1:tc1) voltage
Values:
    0   1.0
        0.25
    1   10.0
        0.5
"""
    )

    result = load_ngspice_rawfile(rawfile)

    waveform = result.waveform("v(r1:tc1)")
    assert result.analysis == "sens"
    assert result.abscissa is not None
    assert result.abscissa.name == "frequency"
    assert waveform.vector_kind == "sensitivity"
    assert waveform.quantity == "sensitivity"
    assert waveform.unit is None
    assert waveform.metadata["sensitivity_vector"] == "v(r1:tc1)"
    assert waveform.metadata["sensitivity_element"] == "r1"
    assert waveform.metadata["sensitivity_parameter"] == "tc1"
    assert result.metadata["vector_kinds"]["v(r1:tc1)"] == "sensitivity"
    assert result.metadata["vector_quantities"]["v(r1:tc1)"] == "sensitivity"
    assert result.metadata["vector_units"]["v(r1:tc1)"] is None
    assert result.metadata["vector_metadata"]["v(r1:tc1)"] == {
        "sensitivity_vector": "v(r1:tc1)",
        "sensitivity_element": "r1",
        "sensitivity_parameter": "tc1",
    }
    np.testing.assert_allclose(waveform.abscissa_data, np.array([1.0, 10.0]))


def test_load_ngspice_rawfile_types_pole_zero_vectors(tmp_path):
    rawfile = tmp_path / "pz.raw"
    rawfile.write_text(
        """Title: pole zero
Plotname: Pole-Zero Analysis
Flags: complex
No. Variables: 2
No. Points: 1
Variables:
    0   v(pole(1)) voltage
    1   v(zero(1)) voltage
Values:
    0   -100.0,25.0
        -10.0,0.0
"""
    )

    result = load_ngspice_rawfile(rawfile)

    pole = result.waveform("v(pole(1))")
    zero = result.waveform("v(zero(1))")
    assert result.analysis == "pz"
    assert result.abscissa is None
    assert "scale_vector" not in result.metadata
    assert pole.vector_kind == "pole"
    assert pole.quantity == "pole"
    assert pole.unit is None
    assert pole.metadata["pole_zero_kind"] == "pole"
    assert zero.vector_kind == "zero"
    assert zero.quantity == "zero"
    assert zero.unit is None
    assert zero.metadata["pole_zero_kind"] == "zero"
    assert result.metadata["vector_kinds"] == {"v(pole(1))": "pole", "v(zero(1))": "zero"}
    assert result.metadata["vector_quantities"] == {"v(pole(1))": "pole", "v(zero(1))": "zero"}
    assert result.metadata["vector_units"] == {"v(pole(1))": None, "v(zero(1))": None}
    assert result.metadata["vector_metadata"] == {
        "v(pole(1))": {"pole_zero_kind": "pole"},
        "v(zero(1))": {"pole_zero_kind": "zero"},
    }
    np.testing.assert_allclose(pole.data, np.array([-100.0 + 25.0j]))
    np.testing.assert_allclose(zero.data, np.array([-10.0 + 0.0j]))


def test_load_ngspice_rawfile_types_distortion_vectors(tmp_path):
    rawfile = tmp_path / "disto.raw"
    rawfile.write_text(
        """Title: distortion
Plotname: DISTORTION Analysis
Flags: real
No. Variables: 3
No. Points: 2
Variables:
    0   frequency frequency
    1   v(out) voltage
    2   i(v1) current
Values:
    0   1000.0
        0.01
        0.001
    1   10000.0
        0.02
        0.002
"""
    )

    result = load_ngspice_rawfile(rawfile, element_names=["V1"])

    voltage = result.waveform("v(out)")
    current = result.waveform("i(V1)")
    assert result.analysis == "disto"
    assert result.abscissa is not None
    assert result.abscissa.name == "frequency"
    assert voltage.vector_kind == "distortion"
    assert voltage.quantity == "voltage"
    assert voltage.unit == "V"
    assert voltage.metadata["distortion_vector"] == "v(out)"
    assert current.vector_kind == "distortion"
    assert current.quantity == "current"
    assert current.unit == "A"
    assert current.metadata["distortion_vector"] == "i(V1)"
    assert result.metadata["vector_kinds"] == {"v(out)": "distortion", "i(V1)": "distortion"}
    assert result.metadata["vector_quantities"] == {"v(out)": "voltage", "i(V1)": "current"}
    assert result.metadata["vector_units"] == {"v(out)": "V", "i(V1)": "A"}
    assert result.metadata["vector_metadata"] == {
        "v(out)": {"distortion_vector": "v(out)"},
        "i(V1)": {"distortion_vector": "i(V1)"},
    }
    np.testing.assert_allclose(voltage.abscissa_data, np.array([1000.0, 10000.0]))
