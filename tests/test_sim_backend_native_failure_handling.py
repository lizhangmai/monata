import subprocess
from typing import Any

import pytest

import monata.sim.backends.ngspice as ngspice_backend
from monata.sim.core import (
    DCSpec,
    FourierSpec,
    LocalExecutor,
    NoiseSpec,
    SimTask,
    TransferFunctionSpec,
)
from monata.sim.backends.ngspice import NgspiceRunner, _control_block
from monata.sim.backends.ngspice_plan import NgspiceTaskPlan
from support.native_backend_cases import _ac_circuit, _dc_circuit, _fourier_circuit
pytestmark = pytest.mark.native

def test_missing_ngspice_returns_failed_result():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
    )

    result = NgspiceRunner(executable="definitely-missing-ngspice").run(task)

    assert result.status == "failed"
    assert result.waveforms == {}
    assert result.sweep_var is None
    assert result.metadata["reason"] == "simulator_missing"
    assert result.error_message is not None
    assert "not found" in result.error_message


def test_native_ngspice_timeout_returns_structured_failed_result(monkeypatch):
    observed = {}

    def fake_run(cmd: list[str], **kwargs: Any):
        observed["cmd"] = cmd
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, (float, int))
        observed["timeout"] = timeout
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=float(timeout),
            output="stdout before timeout",
            stderr="stderr before timeout",
        )

    monkeypatch.setattr("monata.sim.backends.ngspice.shutil.which", lambda _name: "/bin/ngspice")
    monkeypatch.setattr("monata.sim.backends.ngspice.subprocess.run", fake_run)
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        timeout=0.01,
    )

    result = NgspiceRunner().run(task)

    assert observed["timeout"] == 0.01
    assert result.status == "failed"
    assert result.metadata["reason"] == "timeout"
    assert result.metadata["timeout_seconds"] == 0.01
    assert result.metadata["ngspice_stdout"] == "stdout before timeout"
    assert result.metadata["ngspice_stderr"] == "stderr before timeout"
    assert result.error_message is not None
    assert "timed out" in result.error_message


def test_native_ngspice_failure_metadata_preserves_user_reserved_keys():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        osdi_paths=["bad'model.osdi"],
        metadata={
            "simulator": "caller-simulator",
            "elapsed_time": "caller-elapsed",
            "reason": "caller-reason",
            "ngspice": {"caller": "keep"},
        },
    )

    result = NgspiceRunner().run(task)

    assert result.status == "failed"
    assert result.metadata["simulator"] == "caller-simulator"
    assert result.metadata["elapsed_time"] == "caller-elapsed"
    assert result.metadata["reason"] == "caller-reason"
    assert result.metadata["ngspice"] == {"caller": "keep"}
    assert result.metadata["ngspice_2"]["simulator"] == "ngspice-subprocess"
    assert result.metadata["ngspice_2"]["reason"] == "invalid_task"


def test_native_ngspice_rejects_malformed_vector_tokens_before_command_emission():
    cases = [
        SimTask(circuit=_ac_circuit(), analysis_spec=TransferFunctionSpec(output="v(out) foo", input_source="V1")),
        SimTask(circuit=_fourier_circuit(), analysis_spec=FourierSpec(frequency=1000, output="v(out) foo", stop=0.002)),
    ]

    for task in cases:
        result = LocalExecutor(max_workers=1).submit(task).result()
        assert result.status == "failed"
        assert result.metadata["reason"] == "invalid_task"


def test_native_ngspice_backend_error_metadata_includes_exception_details(monkeypatch):
    def fail_render(_circuit):
        raise RuntimeError("boom")

    runner = NgspiceRunner()
    monkeypatch.setattr(runner, "_resolve_executable", lambda: "/bin/true")
    monkeypatch.setattr(ngspice_backend, "render_ngspice", fail_render)
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
    )

    result = runner.run(task)

    assert result.status == "failed"
    assert result.metadata["reason"] == "backend_error"
    assert result.metadata["ngspice"]["exception_type"] == "RuntimeError"
    assert "RuntimeError: boom" in result.metadata["ngspice"]["traceback"]


def test_native_ngspice_backend_error_can_reraise_for_debugging(monkeypatch):
    def fail_render(_circuit):
        raise RuntimeError("boom")

    runner = NgspiceRunner()
    monkeypatch.setattr(runner, "_resolve_executable", lambda: "/bin/true")
    monkeypatch.setattr(ngspice_backend, "render_ngspice", fail_render)
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        backend_options={"raise_backend_exceptions": True},
    )

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(task)


def test_native_ngspice_noise_rejects_task_output_names():
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=NoiseSpec(output_node="out", input_source="V1", start=1, stop=1e3, points=3),
        output_names=["out"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "failed"
    assert result.metadata["reason"] == "invalid_task"
    assert "output_names must be empty" in result.error_message


def test_native_ngspice_noise_missing_ac_source_returns_failed_result():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=NoiseSpec(output_node="in", input_source="V1", start=1, stop=1e3, points=3),
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "failed"
    assert result.metadata["reason"] == "subprocess_failed"
    assert "ac input not found" in result.error_message.lower()


def test_native_ngspice_metadata_preserves_user_reserved_keys():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        metadata={
            "analysis": "caller-analysis",
            "osdi_paths": ["caller.osdi"],
            "simulator": "caller-simulator",
            "ngspice": {"caller": "keep"},
        },
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.metadata["analysis"] == "caller-analysis"
    assert result.metadata["osdi_paths"] == ["caller.osdi"]
    assert result.metadata["simulator"] == "caller-simulator"
    assert result.metadata["ngspice"] == {"caller": "keep"}
    assert result.metadata["ngspice_2"]["analysis"] == "dc"
    assert result.metadata["ngspice_2"]["osdi_paths"] == []
    assert result.metadata["ngspice_2"]["simulator"] == "ngspice-subprocess"


def test_ngspice_task_plan_rejects_removed_wrdata_primary_extraction(tmp_path):
    plan = NgspiceTaskPlan(
        analysis_name="dc",
        output_names=("in",),
        output_vectors=("v(in)",),
        output_requests=(),
        command="dc V1 0 1 0.5",
        osdi_paths=(),
        metadata={},
        extraction="wrdata",  # type: ignore[reportArgumentType]
    )

    with pytest.raises(ValueError, match="unsupported ngspice extraction mode: wrdata"):
        _control_block(plan, tmp_path / "result.dat")


def test_invalid_output_name_returns_failed_result():
    with pytest.raises(ValueError, match="output name"):
        SimTask(
            circuit=_dc_circuit(),
            analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
            output_names=["in\nquit"],
        )


def test_missing_osdi_path_returns_failed_result():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        osdi_paths=["missing-model.osdi"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "failed"
    assert result.metadata["reason"] == "model_missing"
    assert "missing-model.osdi" in result.error_message


@pytest.mark.parametrize("name", ["bad\nmodel.osdi", "bad'model.osdi"])
def test_invalid_osdi_path_returns_failed_result(tmp_path, name):
    osdi = tmp_path / name
    osdi.write_text("")
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        osdi_paths=[osdi],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "failed"
    assert result.metadata["reason"] == "invalid_task"


def test_parser_failure_preserves_ngspice_stdout_stderr(monkeypatch):
    class Proc:
        returncode = 0
        stdout = "stdout evidence"
        stderr = "stderr evidence"

    def fail_parse(*_args, **_kwargs):
        raise ValueError("parse broke")

    monkeypatch.setattr("monata.sim.backends.ngspice.shutil.which", lambda _name: "/bin/ngspice")
    monkeypatch.setattr("monata.sim.backends.ngspice.subprocess.run", lambda *_args, **_kwargs: Proc())
    monkeypatch.setattr("monata.sim.backends.ngspice._parse_output", fail_parse)
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
    )

    result = NgspiceRunner().run(task)

    assert result.status == "failed"
    assert result.metadata["reason"] == "parser_failed"
    assert result.metadata["ngspice_stdout"] == "stdout evidence"
    assert result.metadata["ngspice_stderr"] == "stderr evidence"


def test_parser_failure_can_reraise_for_backend_debugging(monkeypatch):
    class Proc:
        returncode = 0
        stdout = "stdout evidence"
        stderr = "stderr evidence"

    def fail_parse(*_args, **_kwargs):
        raise ValueError("parse broke")

    monkeypatch.setattr("monata.sim.backends.ngspice.shutil.which", lambda _name: "/bin/ngspice")
    monkeypatch.setattr("monata.sim.backends.ngspice.subprocess.run", lambda *_args, **_kwargs: Proc())
    monkeypatch.setattr("monata.sim.backends.ngspice._parse_output", fail_parse)
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        backend_options={"raise_backend_exceptions": True},
    )

    with pytest.raises(ValueError, match="parse broke"):
        NgspiceRunner().run(task)
