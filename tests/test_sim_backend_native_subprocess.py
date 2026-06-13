
import numpy as np
import pytest

from monata.sim.core import (
    ACSpec,
    DCSweep,
    DCSpec,
    DistortionSpec,
    FourierSpec,
    LocalExecutor,
    NoiseSpec,
    OPSpec,
    PoleZeroSpec,
    SensitivitySpec,
    SimTask,
    TranSpec,
    TransferFunctionSpec,
    branch_current_vector,
)
from monata.sim.backends.ngspice import _result_metadata
from monata.sim.backends.ngspice_plan import NgspiceTaskPlan, task_plan as _task_plan
from support.native_backend_cases import (
    _ac_circuit,
    _dc_circuit,
    _dc_dual_sweep_circuit,
    _distortion_circuit,
    _fourier_circuit,
    _tran_circuit,
)
pytestmark = pytest.mark.native

def test_native_ngspice_dc_sweep():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    np.testing.assert_allclose(result.sweep_var, np.array([0.0, 0.5, 1.0]))
    assert "in" in result.waveforms
    np.testing.assert_allclose(result.waveforms["in"], result.sweep_var)
    assert result.analysis_result is not None
    assert result.analysis_result.analysis == "dc"
    assert result.analysis_result.source == "rawfile"
    np.testing.assert_allclose(result.analysis_result.waveform("in").data, result.waveforms["in"])
    assert result.metadata["extraction_preference"] == "rawfile"
    assert result.metadata["extraction"] == "rawfile"
    assert result.metadata["fallback_used"] is False


def test_native_ngspice_dc_dual_sweep_smoke():
    task = SimTask(
        circuit=_dc_dual_sweep_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5, secondary=DCSweep("V2", 0, 1, 0.5)),
        output_names=["out"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert len(result.sweep_var) == 9
    assert "out" in result.waveforms
    assert len(result.waveforms["out"]) == 9
    assert result.metadata["source"] == "V1"
    assert result.metadata["secondary"]["source"] == "V2"


def test_native_ngspice_dc_sweep_source_names_cover_spice_sweep_targets():
    cases = [
        (DCSpec(source="1", start=0, stop=1, step=0.5), "dc V1 0 1 0.5"),
        (DCSpec(source="I1", start=0, stop=1e-3, step=1e-3), "dc I1 0 0.001 0.001"),
        (DCSpec(source="Rload", start=1e3, stop=2e3, step=500), "dc Rload 1000.0 2000.0 500"),
        (DCSpec(source="temp", start=-40, stop=125, step=5), "dc temp -40 125 5"),
    ]

    for spec, command in cases:
        plan = _task_plan(SimTask(circuit=_dc_circuit(), analysis_spec=spec, output_names=["in"]))
        assert plan is not None
        assert plan.command == command


def test_native_ngspice_tran_smoke():
    task = SimTask(
        circuit=_tran_circuit(),
        analysis_spec=TranSpec(step=1e-9, stop=5e-9),
        output_names=["in", "out"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert len(result.sweep_var) > 0
    assert {"in", "out"}.issubset(result.waveforms)
    assert len(result.waveforms["out"]) == len(result.sweep_var)
    assert result.analysis_result is not None
    assert result.analysis_result.analysis == "tran"
    assert result.analysis_result.abscissa is not None
    assert result.analysis_result.abscissa.name == "time"
    np.testing.assert_allclose(result.analysis_result.waveform("out").data, result.waveforms["out"])


def test_native_ngspice_tran_accepts_explicit_vector_requests():
    task = SimTask(
        circuit=_tran_circuit(),
        analysis_spec=TranSpec(step=1e-9, stop=5e-9),
        output_names=[branch_current_vector("V1")],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert "i(V1)" in result.waveforms
    assert result.metadata["output_vectors"] == ["i(V1)"]
    assert result.metadata["output_requests"][0]["vector_kind"] == "branch_current"
    assert result.analysis_result is not None
    waveform = result.analysis_result.waveform("i(v1)")
    assert waveform.vector_kind == "branch_current"
    assert waveform.quantity == "current"
    assert waveform.unit == "A"
    np.testing.assert_allclose(waveform.data, result.waveforms["i(V1)"])


def test_native_ngspice_ac_smoke():
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=ACSpec(start=1, stop=1e6, points=10),
        output_names=["out"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert len(result.sweep_var) > 10
    assert result.sweep_var[0] == pytest.approx(1.0)
    assert result.sweep_var[-1] == pytest.approx(1e6)
    assert "out" in result.waveforms
    assert result.metadata["analysis"] == "ac"
    assert result.metadata["extraction"] == "rawfile"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["output_vectors"] == ["v(out)"]
    assert result.metadata["output_requests"][0]["output_name"] == "out"
    assert result.metadata["output_requests"][0]["vector"] == "v(out)"
    assert result.metadata["output_requests"][0]["raw_vector_name"] == "v(out)"
    assert result.metadata["output_requests"][0]["normalized_name"] == "out"
    assert result.metadata["output_requests"][0]["vector_kind"] == "node_voltage"
    assert result.metadata["output_requests"][0]["transform"] == "mag"
    assert result.metadata["vector_kinds"]["out"] == "node_voltage"
    assert result.analysis_result is not None
    assert result.analysis_result.waveform("out").vector_kind == "node_voltage"
    assert result.metadata["ngspice"]["analysis"] == "ac"
    assert result.metadata["ngspice"]["output_vectors"] == ["v(out)"]
    assert result.metadata["extraction_preference"] == "rawfile"


def test_native_ngspice_ac_python_side_transforms_use_raw_complex_vector():
    outputs = ["v(out)", "mag(v(out))", "phase(v(out))", "real(v(out))", "imag(v(out))", "db(v(out))"]
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=ACSpec(start=1, stop=1e6, points=10),
        output_names=outputs,
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.metadata["extraction"] == "rawfile"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["output_vectors"] == ["v(out)"] * len(outputs)
    raw = result.waveforms["v(out)"]
    np.testing.assert_allclose(result.waveforms["mag(v(out))"], np.abs(raw))
    np.testing.assert_allclose(result.waveforms["phase(v(out))"], np.angle(raw))
    np.testing.assert_allclose(result.waveforms["real(v(out))"], np.real(raw))
    np.testing.assert_allclose(result.waveforms["imag(v(out))"], np.imag(raw))
    with np.errstate(divide="ignore"):
        np.testing.assert_allclose(result.waveforms["db(v(out))"], 20 * np.log10(np.abs(raw)))
    assert result.analysis_result is not None
    assert result.analysis_result.waveform("mag(v(out))").raw_vector_name == "v(out)"
    assert result.analysis_result.waveform("phase(v(out))").raw_vector_name == "v(out)"
    assert result.analysis_result.waveform("phase(v(out))").quantity == "phase"
    assert result.analysis_result.waveform("phase(v(out))").unit == "rad"
    assert result.analysis_result.waveform("db(v(out))").quantity == "gain_db"
    assert result.analysis_result.waveform("db(v(out))").unit == "dB"
    assert "phase(v(out))" not in result.analysis_result.voltages


def test_native_ngspice_op_smoke():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=OPSpec(),
        output_names=["in"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is None
    assert "in" in result.waveforms
    np.testing.assert_allclose(result.waveforms["in"], np.array([0.0]))
    assert result.metadata["analysis"] == "op"
    assert result.metadata["output_vectors"] == ["v(in)"]
    assert result.metadata["extraction"] == "stdout-print"
    assert result.metadata["extraction_preference"] == "rawfile"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_reason"] == "rawfile_not_enabled_for_analysis"
    assert result.analysis_result is not None
    assert result.analysis_result.analysis == "op"
    assert result.analysis_result.source == "stdout-print"
    np.testing.assert_allclose(result.analysis_result.waveform("in").data, result.waveforms["in"])


def test_native_ngspice_noise_smoke():
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=NoiseSpec(output_node="out", input_source="V1", start=1, stop=1e3, points=3),
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert len(result.sweep_var) > 0
    assert {"onoise_spectrum", "inoise_spectrum"}.issubset(result.waveforms)
    assert len(result.waveforms["onoise_spectrum"]) == len(result.sweep_var)
    assert result.metadata["analysis"] == "noise"
    assert result.metadata["output_vectors"] == ["onoise_spectrum", "inoise_spectrum"]
    assert result.metadata["noise_output_node"] == "out"
    assert result.metadata["noise_input_source"] == "V1"
    assert result.metadata["variation"] == "dec"
    assert result.metadata["extraction"] == "stdout-print"
    assert result.metadata["noise_totals"]["onoise_total"] > 0
    assert result.metadata["noise_totals"]["inoise_total"] > 0
    assert result.metadata["extraction_preference"] == "rawfile"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_reason"] == "rawfile_not_enabled_for_analysis"
    assert result.analysis_result is not None
    assert result.analysis_result.analysis == "noise"
    assert result.analysis_result.abscissa is not None
    assert result.analysis_result.abscissa.name == "frequency"
    np.testing.assert_allclose(
        result.analysis_result.waveform("onoise_spectrum").data,
        result.waveforms["onoise_spectrum"],
    )


def test_native_ngspice_sensitivity_smoke():
    task = SimTask(circuit=_ac_circuit(), analysis_spec=SensitivitySpec(output="v(out)"))

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.metadata["analysis"] == "sens"
    assert result.metadata["extraction"] == "rawfile"
    assert result.waveforms
    assert set(result.metadata["vector_kinds"].values()) == {"sensitivity"}
    assert set(result.metadata["vector_quantities"].values()) == {"sensitivity"}
    assert set(result.metadata["vector_units"]) == set(result.waveforms)
    assert all(unit is None for unit in result.metadata["vector_units"].values())
    assert result.analysis_result is not None
    waveform = result.analysis_result.waveform("v(r1:bv_max)")
    assert waveform.vector_kind == "sensitivity"
    assert waveform.quantity == "sensitivity"
    assert waveform.unit is None
    assert waveform.metadata["sensitivity_element"] == "r1"
    assert waveform.metadata["sensitivity_parameter"] == "bv_max"


def test_native_ngspice_ac_sensitivity_uses_frequency_abscissa():
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=SensitivitySpec(output="v(out)", start=1, stop=1000, points=3),
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert result.metadata["analysis"] == "sens"
    assert result.metadata["variation"] == "dec"
    assert result.metadata["start"] == 1
    assert result.metadata["stop"] == 1000
    assert result.metadata["points"] == 3
    assert result.analysis_result is not None
    assert result.analysis_result.abscissa is not None
    assert result.analysis_result.abscissa.name == "frequency"
    assert result.analysis_result.abscissa.unit == "Hz"
    assert not np.iscomplexobj(result.analysis_result.frequency)
    waveform = result.analysis_result.waveform("v(r1:bv_max)")
    assert waveform.vector_kind == "sensitivity"
    assert waveform.abscissa_name == "frequency"
    np.testing.assert_allclose(waveform.abscissa_data, result.analysis_result.frequency)


def test_native_ngspice_pole_zero_smoke():
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=PoleZeroSpec(input_pos="in", input_neg="0", output_pos="out", output_neg="0"),
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.metadata["analysis"] == "pz"
    assert result.metadata["extraction"] == "rawfile"
    assert result.waveforms
    assert result.sweep_var is None
    assert "v(pole(1))" in result.waveforms
    assert set(result.metadata["vector_kinds"].values()) <= {"pole", "zero"}
    assert set(result.metadata["vector_quantities"].values()) <= {"pole", "zero"}
    assert set(result.metadata["vector_units"]) == set(result.waveforms)
    assert all(unit is None for unit in result.metadata["vector_units"].values())
    assert result.analysis_result is not None
    assert result.analysis_result.abscissa is None
    assert "v(pole(1))" in result.analysis_result.waveforms
    kinds = {waveform.vector_kind for waveform in result.analysis_result.waveforms.values()}
    assert kinds <= {"pole", "zero"}
    waveform = next(iter(result.analysis_result.waveforms.values()))
    assert waveform.quantity == waveform.vector_kind
    assert waveform.unit is None
    assert waveform.metadata["pole_zero_kind"] == waveform.vector_kind


def test_native_ngspice_distortion_smoke():
    task = SimTask(circuit=_distortion_circuit(), analysis_spec=DistortionSpec(start=1, stop=1e3, points=3))

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert result.metadata["analysis"] == "disto"
    assert result.metadata["extraction"] == "rawfile"
    assert result.waveforms
    assert result.metadata["vector_kinds"] == {
        "v(in)": "distortion",
        "v(out)": "distortion",
        "i(v1)": "distortion",
    }
    assert result.metadata["vector_quantities"] == {
        "v(in)": "voltage",
        "v(out)": "voltage",
        "i(v1)": "current",
    }
    assert result.metadata["vector_units"] == {
        "v(in)": "V",
        "v(out)": "V",
        "i(v1)": "A",
    }
    assert result.analysis_result is not None
    assert result.analysis_result.abscissa is not None
    assert result.analysis_result.abscissa.name == "frequency"
    assert result.analysis_result.abscissa.unit == "Hz"
    assert not np.iscomplexobj(result.analysis_result.frequency)
    out = result.analysis_result.waveform("v(out)")
    branch = result.analysis_result.waveform("i(v1)")
    assert out.vector_kind == "distortion"
    assert out.quantity == "voltage"
    assert out.unit == "V"
    assert out.metadata["distortion_vector"] == "v(out)"
    assert branch.quantity == "current"
    assert branch.unit == "A"


def test_native_ngspice_transfer_function_smoke():
    task = SimTask(circuit=_ac_circuit(), analysis_spec=TransferFunctionSpec(output="v(out)", input_source="V1"))

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.metadata["analysis"] == "tf"
    assert result.metadata["extraction"] == "stdout-print"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["output_vectors"] == ["transfer_function", "input_resistance", "output_resistance"]
    assert result.metadata["output_requests"][0]["vector"] == "transfer_function"
    assert result.metadata["vector_kinds"] == {
        "transfer_function": "transfer_function",
        "input_resistance": "transfer_function",
        "output_resistance": "transfer_function",
    }
    assert result.metadata["vector_quantities"] == {
        "transfer_function": "gain",
        "input_resistance": "resistance",
        "output_resistance": "resistance",
    }
    assert result.metadata["vector_units"] == {
        "input_resistance": "Ohm",
        "output_resistance": "Ohm",
    }
    assert {"transfer_function", "input_resistance", "output_resistance"}.issubset(result.waveforms)
    np.testing.assert_allclose(result.waveforms["transfer_function"], np.array([1.0]))
    np.testing.assert_allclose(result.waveforms["output_resistance"], np.array([1000.0]))
    assert result.waveforms["input_resistance"][0] > 1e12
    assert result.analysis_result is not None
    transfer = result.analysis_result.waveform("transfer_function")
    assert transfer.vector_kind == "transfer_function"
    assert transfer.metadata == {"tf_output_vector": "v(out)", "tf_input_source": "V1"}
    input_resistance = result.analysis_result.waveform("input_resistance")
    assert input_resistance.quantity == "resistance"
    assert input_resistance.unit == "Ohm"
    assert input_resistance.metadata == {"tf_input_source": "V1"}
    assert result.analysis_result.waveform("output_resistance").metadata == {"tf_output_vector": "v(out)"}


def test_native_ngspice_fourier_smoke():
    task = SimTask(
        circuit=_fourier_circuit(),
        analysis_spec=FourierSpec(frequency=1000, output="v(out)", stop=0.002, step=1e-5),
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.metadata["analysis"] == "four"
    assert result.metadata["extraction"] == "stdout-print"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["fourier_output_vector"] == "v(out)"
    assert {"harmonic", "frequency", "fourier_magnitude", "fourier_phase"}.issubset(result.waveforms)
    assert result.metadata["fourier_harmonic_count"] == len(result.waveforms["harmonic"])
    if "fourier_thd_percent" in result.metadata:
        assert result.metadata["fourier_thd_percent"] >= 0
    if "fourier_grid_size" in result.metadata:
        assert result.metadata["fourier_grid_size"] > 0
    if "fourier_interpolation_degree" in result.metadata:
        assert result.metadata["fourier_interpolation_degree"] >= 0
    if "fourier_period_count" in result.metadata:
        assert result.metadata["fourier_period_count"] >= 1
    assert result.metadata["output_vectors"] == list(result.waveforms)
    assert result.metadata["output_requests"][2]["vector"] == "fourier_magnitude"
    assert result.metadata["vector_kinds"] == {
        name: "fourier_component"
        for name in result.waveforms
    }
    assert result.metadata["vector_quantities"] == {
        name: quantity
        for name, quantity in {
            "harmonic": "harmonic",
            "frequency": "frequency",
            "fourier_magnitude": "magnitude",
            "fourier_phase": "phase",
            "fourier_normalized_magnitude": "normalized_magnitude",
            "fourier_normalized_phase": "normalized_phase",
        }.items()
        if name in result.waveforms
    }
    assert result.metadata["vector_units"] == {
        name: unit
        for name, unit in {
            "frequency": "Hz",
            "fourier_phase": "deg",
            "fourier_normalized_phase": "deg",
        }.items()
        if name in result.waveforms
    }
    assert len(result.waveforms["fourier_magnitude"]) >= 2
    if "fourier_normalized_magnitude" in result.waveforms:
        assert result.waveforms["fourier_normalized_magnitude"][1] == pytest.approx(1.0)
    assert result.analysis_result is not None
    assert result.analysis_result.abscissa is not None
    assert result.analysis_result.abscissa.name == "frequency"
    if "fourier_thd_percent" in result.metadata:
        assert result.analysis_result.metadata["fourier_thd_percent"] == result.metadata["fourier_thd_percent"]
    magnitude = result.analysis_result.waveform("fourier_magnitude")
    assert magnitude.vector_kind == "fourier_component"
    assert magnitude.metadata == {"fourier_output_vector": "v(out)", "fourier_frequency": 1000}
    assert result.analysis_result.waveform("fourier_phase").quantity == "phase"
    assert result.analysis_result.waveform("fourier_phase").unit == "deg"
    if "fourier_normalized_magnitude" in result.waveforms:
        assert result.analysis_result.waveform("fourier_normalized_magnitude").quantity == "normalized_magnitude"
    if "fourier_normalized_phase" in result.waveforms:
        assert result.analysis_result.waveform("fourier_normalized_phase").unit == "deg"


def test_native_ngspice_success_metadata_merge_contract():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        metadata={
            "analysis": "caller-analysis",
            "ngspice": {"caller": "keep"},
        },
    )
    plan = NgspiceTaskPlan(
        analysis_name="dc",
        output_names=("in",),
        output_vectors=("v(in)",),
        output_requests=(),
        command="dc V1 0 1 0.5",
        osdi_paths=(),
        metadata={
            "analysis": "dc",
            "output_vectors": ["v(in)"],
            "extraction": "rawfile",
        },
    )

    metadata = _result_metadata(
        task,
        plan,
        1.25,
        {
            "analysis": "parser-analysis",
            "backend_only": "diagnostic",
            "structured_mutations": [{"target": "R1.R", "value": 1000}],
        },
    )

    assert metadata["analysis"] == "caller-analysis"
    assert metadata["ngspice"] == {"caller": "keep"}
    assert metadata["output_vectors"] == ["v(in)"]
    assert metadata["structured_mutations"] == [{"target": "R1.R", "value": 1000}]
    assert "backend_only" not in metadata
    assert metadata["ngspice_2"]["analysis"] == "parser-analysis"
    assert metadata["ngspice_2"]["backend_only"] == "diagnostic"
    assert metadata["ngspice_2"]["extraction"] == "rawfile"
    assert metadata["ngspice_2"]["simulator"] == "ngspice-subprocess"
    assert metadata["ngspice_2"]["elapsed_time"] == 1.25
    assert metadata["ngspice_2"]["outputs"] == ["in"]


def test_backend_specific_analysis_spec_strings_return_failed_result():
    cases = [
        SimTask(
            circuit=_dc_circuit(),
            analysis_spec=DCSpec(source="V 1", start=0, stop=1, step=0.5),
            output_names=["in"],
        ),
        SimTask(
            circuit=_ac_circuit(),
            analysis_spec=NoiseSpec(output_node="out bad", input_source="V1", start=1, stop=10, points=3),
        ),
        SimTask(
            circuit=_ac_circuit(),
            analysis_spec=TransferFunctionSpec(output="v(out) bad", input_source="V1"),
        ),
        SimTask(
            circuit=_fourier_circuit(),
            analysis_spec=FourierSpec(frequency=1000, output="v(out) bad", stop=0.002),
        ),
    ]

    for task in cases:
        result = LocalExecutor(max_workers=1).submit(task).result()
        assert result.status == "failed"
        assert result.metadata["reason"] == "invalid_task"
