from pathlib import Path

import numpy as np
import pytest

from monata.sim.core import (
    ACSpec,
    DCSpec,
    LocalExecutor,
    OPSpec,
    SimTask,
    TranSpec,
    expression_vector,
    voltage_vector,
)
from support.native_backend_cases import _ac_circuit, _dc_circuit, _tran_circuit
pytestmark = pytest.mark.native

def test_native_ngspice_binary_rawfile_matches_ascii_dc_sweep():
    ascii_task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        backend_options={"rawfile_format": "ascii"},
    )
    binary_task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        backend_options={"rawfile_format": "binary"},
    )

    executor = LocalExecutor(max_workers=1)
    ascii_result = executor.submit(ascii_task).result()
    binary_result = executor.submit(binary_task).result()

    assert ascii_result.status == "ok", ascii_result.error_message
    assert binary_result.status == "ok", binary_result.error_message
    assert ascii_result.metadata["rawfile_format"] == "ascii"
    assert binary_result.metadata["rawfile_format"] == "binary"
    np.testing.assert_allclose(binary_result.sweep_var, ascii_result.sweep_var)
    np.testing.assert_allclose(binary_result.waveforms["in"], ascii_result.waveforms["in"])
    assert binary_result.analysis_result is not None
    np.testing.assert_allclose(binary_result.analysis_result.waveform("in").data, ascii_result.waveforms["in"])


def test_native_ngspice_tran_measure_results_preserve_waveforms():
    circuit = _tran_circuit()
    circuit.measure("tran", "vout_2n", "FIND v(out) AT=2n")
    task = SimTask(
        circuit=circuit,
        analysis_spec=TranSpec(step=1e-9, stop=5e-9),
        output_names=["out"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert "out" in result.waveforms
    assert len(result.waveforms["out"]) == len(result.sweep_var)
    assert result.measures["vout_2n"].analysis == "tran"
    assert result.measures["vout_2n"].value is not None
    assert result.metadata["measures"]["vout_2n"]["source"] == "simulator_measure"
    assert result.metadata["ngspice"]["measures"]["vout_2n"]["value"] == result.measures.value("vout_2n")


def test_native_ngspice_persists_requested_artifacts(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "tasks" / "task-0000"
    task = SimTask(
        circuit=_tran_circuit(),
        analysis_spec=TranSpec(step=1e-9, stop=5e-9),
        output_names=["out"],
        artifacts=artifact_dir,
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    artifacts = result.metadata["artifacts"]
    assert artifacts["directory"] == str(artifact_dir)
    assert Path(artifacts["files"]["netlist"]).read_text().startswith("tran sanity")
    assert Path(artifacts["files"]["rawfile"]).is_file()
    assert Path(artifacts["files"]["wrdata"]).is_file()
    assert Path(artifacts["files"]["stdout"]).is_file()
    assert Path(artifacts["files"]["stderr"]).is_file()
    metadata = Path(artifacts["files"]["metadata"]).read_text()
    assert '"schema": "monata-simulation-artifacts-v1"' in metadata
    assert '"simulator": "ngspice-subprocess"' in metadata


def test_native_ngspice_multiple_and_failed_measure_results():
    circuit = _tran_circuit()
    circuit.measure("tran", "vout_1n", "FIND v(out) AT=1n")
    circuit.measure("tran", "never_crossed", "WHEN v(out)=2")
    task = SimTask(
        circuit=circuit,
        analysis_spec=TranSpec(step=1e-9, stop=5e-9),
        output_names=["out"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.measures["vout_1n"].value is not None
    assert result.measures["never_crossed"].value is None
    assert result.measures["never_crossed"].reason in {"measure_failed", "measure_missing"}
    assert result.metadata["measures"]["never_crossed"]["reason"] == result.measures["never_crossed"].reason


def test_native_ngspice_ac_measure_results():
    circuit = _ac_circuit()
    circuit.measure("ac", "gain_1k", "FIND v(out) AT=1k")
    task = SimTask(
        circuit=circuit,
        analysis_spec=ACSpec(start=1, stop=1e6, points=20),
        output_names=["out"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.measures["gain_1k"].analysis == "ac"
    assert result.measures["gain_1k"].value is not None
    assert result.measures["gain_1k"].raw is not None
    assert "gain_1k" in result.measures["gain_1k"].raw
    assert result.metadata["measures"]["gain_1k"]["raw"] == result.measures["gain_1k"].raw
    assert result.metadata["extraction"] == "rawfile"
    assert result.metadata["fallback_used"] is False
    assert result.analysis_result is not None
    assert result.analysis_result.analysis == "ac"
    assert result.analysis_result.abscissa is not None
    assert result.analysis_result.abscissa.name == "frequency"
    np.testing.assert_allclose(result.analysis_result.waveform("out").data, result.waveforms["out"])


def test_native_ngspice_ac_raw_complex_vector_uses_rawfile():
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=ACSpec(start=1, stop=1e6, points=10),
        output_names=["v(out)"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert "v(out)" in result.waveforms
    assert result.metadata["extraction"] == "rawfile"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["output_vectors"] == ["v(out)"]
    assert result.metadata["output_requests"][0]["output_name"] == "v(out)"
    assert result.metadata["output_requests"][0]["vector"] == "v(out)"
    assert result.metadata["output_requests"][0]["raw_vector_name"] == "v(out)"
    assert result.metadata["output_requests"][0]["vector_kind"] == "node_voltage"
    assert np.iscomplexobj(result.waveforms["v(out)"])
    assert result.analysis_result is not None
    waveform = result.analysis_result.waveform("v(out)")
    assert waveform.vector_kind == "node_voltage"
    assert waveform.raw_vector_name == "v(out)"
    np.testing.assert_allclose(waveform.data, result.waveforms["v(out)"])


def test_native_ngspice_ac_expression_vector_uses_rawfile():
    differential = voltage_vector("in", "out")
    expression = expression_vector("v(in)-v(out)")
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=ACSpec(start=1, stop=1e3, points=3),
        output_names=[differential, expression],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert result.metadata["extraction"] == "rawfile"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["output_vectors"] == [differential, expression]
    assert result.metadata["output_requests"][0]["vector_kind"] == "differential_voltage"
    assert result.metadata["output_requests"][1]["vector_kind"] == "expression"
    assert result.metadata["vector_raw_names"] == {
        differential: differential,
        expression: expression,
    }
    assert result.metadata["vector_kinds"] == {
        differential: "differential_voltage",
        expression: "expression",
    }
    assert result.metadata["vector_units"][differential] == "V"
    assert np.iscomplexobj(result.waveforms[differential])
    assert np.iscomplexobj(result.waveforms[expression])
    np.testing.assert_allclose(result.waveforms[expression], result.waveforms[differential])
    assert result.analysis_result is not None
    assert result.analysis_result.waveform(differential).vector_kind == "differential_voltage"
    assert result.analysis_result.waveform(expression).vector_kind == "expression"
    assert set(result.analysis_result.expressions) == {expression}
    np.testing.assert_allclose(result.analysis_result.waveform(expression).data, result.waveforms[differential])


def test_native_ngspice_ac_branch_current_uses_rawfile():
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=ACSpec(start=1, stop=1e6, points=10),
        output_names=["i(v1)"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    assert "i(v1)" in result.waveforms
    assert result.metadata["extraction"] == "rawfile"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["output_vectors"] == ["i(v1)"]
    assert result.metadata["output_requests"][0]["vector_kind"] == "branch_current"
    assert np.iscomplexobj(result.waveforms["i(v1)"])
    assert result.analysis_result is not None
    waveform = result.analysis_result.waveform("i(v1)")
    assert waveform.vector_kind == "branch_current"
    assert waveform.raw_vector_name == "i(v1)"
    np.testing.assert_allclose(waveform.data, result.waveforms["i(v1)"])


def test_native_ngspice_op_without_output_names_writes_all_rawfile_vectors():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=OPSpec(),
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is None
    assert result.metadata["analysis"] == "op"
    assert result.metadata["write_all_vectors"] is True
    assert result.metadata["extraction"] == "rawfile"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["output_vectors"] == ["v(in)", "i(v1)"]
    assert [request["vector_kind"] for request in result.metadata["output_requests"]] == [
        "node_voltage",
        "branch_current",
    ]
    assert result.analysis_result is not None
    assert result.analysis_result.source == "rawfile"
    assert result.analysis_result.waveform("in").vector_kind == "node_voltage"
    assert result.analysis_result.waveform("i(v1)").vector_kind == "branch_current"


def test_native_ngspice_dc_without_output_names_writes_all_rawfile_vectors():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.metadata["analysis"] == "dc"
    assert result.metadata["write_all_vectors"] is True
    assert result.metadata["output_vectors"] == ["v(in)", "i(v1)"]
    assert [request["vector"] for request in result.metadata["output_requests"]] == ["v(in)", "i(v1)"]
    assert [request["vector_kind"] for request in result.metadata["output_requests"]] == [
        "node_voltage",
        "branch_current",
    ]
    assert result.analysis_result is not None
    assert result.analysis_result.waveform("in").vector_kind == "node_voltage"
    assert result.analysis_result.waveform("i(v1)").vector_kind == "branch_current"
