import shlex
from pathlib import Path

import numpy as np
import pytest

from monata.netlist import Circuit
from monata.sim.analysis_spec import DCSpec, TranSpec
from monata.sim.backends.ngspice_shared import (
    NgspiceInitData,
    NgspiceSharedCallbacks,
    NgspiceSharedRunner,
    NgspiceSharedSession,
    _execute_plan,
)
from monata.sim.backends.ngspice_shared_ffi import library_candidates
from monata.sim.backends.ngspice_plan import NgspiceTaskPlan
from monata.sim.core import LocalExecutor
from monata.sim.results import SimResult
from monata.sim.task import SimTask


def _rc_circuit() -> Circuit:
    circuit = Circuit("shared rc")
    circuit.voltage("1", "in", "0", 1)
    circuit.resistor("1", "in", "out", "1k")
    circuit.resistor("2", "out", "0", "1k")
    return circuit


def test_shared_runner_missing_library_returns_structured_failure():
    task = SimTask(
        circuit=_rc_circuit(),
        analysis_spec=TranSpec(stop=1e-6),
        simulator="ngspice-shared",
        output_names=["out"],
    )

    result = NgspiceSharedRunner(library="definitely-missing-libngspice-for-monata").run(task)

    assert result.status == "failed"
    assert result.metadata["simulator"] == "ngspice-shared"
    assert result.metadata["reason"] == "simulator_missing"
    assert result.error_message is not None
    assert "libngspice" in result.error_message


def test_shared_library_candidates_include_common_runtime_soname():
    assert "libngspice.so.0" in library_candidates(None)


def test_shared_runner_rejects_non_native_circuit_before_opening_session():
    class FailingSession:
        def __init__(self, library):
            raise AssertionError("session should not be opened")

    task = SimTask(
        circuit=object(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        simulator="ngspice-shared",
        output_names=["out"],
    )

    result = NgspiceSharedRunner(session_factory=FailingSession).run(task)

    assert result.status == "failed"
    assert result.metadata["reason"] == "invalid_circuit"
    assert result.error_message == "shared ngspice execution requires a monata.netlist.Circuit"


def test_shared_runner_backend_error_can_reraise_for_debugging():
    class FailingSession:
        stderr = ""

        def __init__(self, library):
            self.library = library

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def load_circuit(self, _circuit):
            raise RuntimeError("boom")

    task = SimTask(
        circuit=_rc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        simulator="ngspice-shared",
        output_names=["out"],
        backend_options={"raise_backend_exceptions": True},
    )

    with pytest.raises(RuntimeError, match="boom"):
        NgspiceSharedRunner(session_factory=FailingSession).run(task)


def test_shared_runner_uses_session_command_plan_and_parser(monkeypatch):
    commands = []
    loaded = []

    class FakeSession:
        stderr = ""

        def __init__(self, library):
            self.library = library

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def load_circuit(self, circuit):
            loaded.append(circuit)

        def command(self, command):
            commands.append(command)
            return f"ran {command}"

    def fake_parse_output(output_path, stdout, plan):
        assert "dc v1 0 1 0.5" in stdout
        assert output_path.name == "result.raw"
        return (
            np.array([0.0, 0.5, 1.0]),
            {"out": np.array([0.0, 0.25, 0.5])},
            None,
            {"extraction": "rawfile"},
        )

    monkeypatch.setattr("monata.sim.backends.ngspice_shared._parse_output", fake_parse_output)
    task = SimTask(
        circuit=_rc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        simulator="ngspice-shared",
        output_names=["out"],
    )

    result = NgspiceSharedRunner(session_factory=FakeSession).run(task)

    assert result.status == "ok"
    assert result.metadata["simulator"] == "ngspice-shared"
    assert ".end" in loaded[0]
    assert commands[:2] == ["set wr_singlescale", "dc v1 0 1 0.5"]
    assert any(command.startswith("write ") and "v(out)" in command for command in commands)
    np.testing.assert_allclose(result.waveforms["out"], np.array([0.0, 0.25, 0.5]))


def test_shared_runner_injects_task_directives_without_duplicate_end(monkeypatch):
    loaded = []

    class FakeSession:
        stderr = ""

        def __init__(self, library):
            self.library = library

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def load_circuit(self, circuit):
            loaded.append(circuit)

        def command(self, command):
            return f"ran {command}"

    def fake_parse_output(output_path, stdout, plan):
        return (
            np.array([0.0, 0.5, 1.0]),
            {"out": np.array([0.0, 0.25, 0.5])},
            None,
            {"extraction": "rawfile"},
        )

    monkeypatch.setattr("monata.sim.backends.ngspice_shared._parse_output", fake_parse_output)
    circuit = _rc_circuit()
    circuit.param("rload", "1k")
    task = SimTask(
        circuit=circuit,
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        simulator="ngspice-shared",
        output_names=["out"],
        param_overrides={"rload": "2k"},
    )

    result = NgspiceSharedRunner(session_factory=FakeSession).run(task)

    assert result.status == "ok", result.error_message
    deck = loaded[0]
    assert ".param rload=2k" in deck
    assert deck.strip().endswith(".end")
    assert deck.splitlines().count(".end") == 1


def test_shared_execute_plan_emits_rawfile_commands_in_order(tmp_path):
    output_path = tmp_path / "result.raw"
    osdi_path = tmp_path / "model.osdi"
    osdi_path.write_text("", encoding="utf-8")
    plan = NgspiceTaskPlan(
        analysis_name="dc",
        output_names=("out",),
        output_vectors=("v(out)",),
        output_requests=(),
        command="dc V1 0 1 0.5",
        osdi_paths=(osdi_path,),
        metadata={"rawfile_format": "binary"},
    )

    class FakeSession:
        def __init__(self):
            self.commands = []

        def command(self, command):
            self.commands.append(command)
            return f"ran {command}"

    session = FakeSession()

    stdout = _execute_plan(session, plan, output_path)

    assert session.commands == [
        "set wr_singlescale",
        f"pre_osdi '{osdi_path}'",
        "dc v1 0 1 0.5",
        "set filetype=binary",
        f"write '{output_path}' v(out)",
        f"wrdata '{output_path.with_suffix('.dat')}' v(out)",
    ]
    assert stdout == "\n".join(f"ran {command}" for command in session.commands)


def test_shared_runner_persists_requested_artifacts(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts" / "tasks" / "task-0000"

    class FakeSession:
        stderr = ""

        def __init__(self, library):
            self.library = library

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def load_circuit(self, circuit):
            self.circuit = circuit

        def command(self, command):
            if command.startswith(("write ", "wrdata ")):
                Path(shlex.split(command)[1]).write_text(command, encoding="utf-8")
            return f"ran {command}"

    def fake_parse_output(output_path, stdout, plan):
        return (
            np.array([0.0, 0.5, 1.0]),
            {"out": np.array([0.0, 0.25, 0.5])},
            None,
            {"extraction": "rawfile"},
        )

    monkeypatch.setattr("monata.sim.backends.ngspice_shared._parse_output", fake_parse_output)
    task = SimTask(
        circuit=_rc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        simulator="ngspice-shared",
        output_names=["out"],
        artifacts=artifact_dir,
    )

    result = NgspiceSharedRunner(session_factory=FakeSession).run(task)

    assert result.status == "ok", result.error_message
    artifacts = result.metadata["artifacts"]
    assert artifacts["directory"] == str(artifact_dir)
    assert Path(artifacts["files"]["netlist"]).read_text().startswith("shared rc")
    assert Path(artifacts["files"]["rawfile"]).is_file()
    assert Path(artifacts["files"]["wrdata"]).is_file()
    assert Path(artifacts["files"]["stdout"]).is_file()
    assert Path(artifacts["files"]["stderr"]).is_file()
    metadata = Path(artifacts["files"]["metadata"]).read_text()
    assert '"schema": "monata-simulation-artifacts-v1"' in metadata
    assert '"simulator": "ngspice-shared"' in metadata


def test_shared_session_exposes_callback_handler_and_event_records(monkeypatch):
    class FakeLib:
        def ngSpice_Init(self, send_char, send_stat, exit_cb, send_data, send_init_data, bg_running, handle):
            self.send_data = send_data
            self.send_init_data = send_init_data
            return 0

        def ngSpice_Init_Sync(self, get_vsrc_data, get_isrc_data, get_sync_data, ngspice_id, handle):
            self.get_vsrc_data = get_vsrc_data
            self.get_isrc_data = get_isrc_data
            return 0

        def ngSpice_Command(self, command):
            return 0

        def ngSpice_running(self):
            return False

    class CallbackRecorder(NgspiceSharedCallbacks):
        def __init__(self):
            self.messages = []
            self.values = []
            self.init = []
            self.vsrc_requests = []
            self.isrc_requests = []

        def send_char(self, session, message, ngspice_id):
            self.messages.append((message, ngspice_id))
            return 0

        def send_data(self, session, values, vector_count, ngspice_id):
            self.values.append((values, vector_count, ngspice_id))
            return 0

        def send_init_data(self, session, data, ngspice_id):
            self.init.append((data, ngspice_id))
            return 0

        def get_vsrc_data(self, session, time_value, node, ngspice_id):
            self.vsrc_requests.append((time_value, node, ngspice_id))
            return 1.25

        def get_isrc_data(self, session, time_value, node, ngspice_id):
            self.isrc_requests.append((time_value, node, ngspice_id))
            return -0.25

    fake = FakeLib()
    monkeypatch.setattr("monata.sim.backends.ngspice_shared_session._load_library", lambda ffi, library: fake)
    callbacks = CallbackRecorder()

    session = NgspiceSharedSession(callbacks=callbacks, enable_data_callbacks=True, enable_sync_callbacks=True)
    ffi = session._ffi

    assert fake.send_data != ffi.NULL
    assert fake.send_init_data != ffi.NULL
    assert fake.get_vsrc_data != ffi.NULL

    session._send_char(ffi.new("char[]", b"stdout hello"), 7, ffi.NULL)
    assert session.stdout == "hello"
    assert callbacks.messages == [("stdout hello", 7)]
    assert session.callback_events[-1].kind == "send_char"

    value_name = ffi.new("char[]", b"v(out)")
    value = ffi.new("vecvalues *", {"name": value_name, "creal": 0.5, "cimag": 0.0, "is_scale": False, "is_complex": False})
    values = ffi.new("pvecvalues[]", [value])
    payload = ffi.new("vecvaluesall *", {"veccount": 1, "vecindex": 0, "vecsa": values})
    session._send_data(payload, 1, 9, ffi.NULL)
    assert callbacks.values == [({"v(out)": 0.5}, 1, 9)]
    assert session.data_events == ({"v(out)": 0.5},)

    vec_name = ffi.new("char[]", b"time")
    vec = ffi.new("vecinfo *", {"number": 0, "vecname": vec_name, "is_real": True})
    vecs = ffi.new("pvecinfo[]", [vec])
    init_payload = ffi.new(
        "vecinfoall *",
        {
            "name": ffi.new("char[]", b"tran1"),
            "title": ffi.new("char[]", b"callback test"),
            "date": ffi.new("char[]", b"today"),
            "type": ffi.new("char[]", b"tran"),
            "veccount": 1,
            "vecs": vecs,
        },
    )
    session._send_init_data(init_payload, 3, ffi.NULL)
    assert isinstance(callbacks.init[0][0], NgspiceInitData)
    assert callbacks.init[0][0].vectors[0].name == "time"

    voltage = ffi.new("double *", 0.0)
    session._get_vsrc_data(voltage, 2e-9, ffi.new("char[]", b"ext"), 5, ffi.NULL)
    assert voltage[0] == pytest.approx(1.25)
    assert callbacks.vsrc_requests == [(2e-9, "ext", 5)]

    current = ffi.new("double *", 0.0)
    session._get_isrc_data(current, 3e-9, ffi.new("char[]", b"sense"), 6, ffi.NULL)
    assert current[0] == pytest.approx(-0.25)
    assert callbacks.isrc_requests == [(3e-9, "sense", 6)]


def test_shared_session_exposes_ngspice_control_command_wrappers():
    class CommandRecorder(NgspiceSharedSession):
        def __init__(self):
            self.commands = []

        def command(self, command):
            self.commands.append(command)
            return f"ran {command}"

    session = CommandRecorder()

    assert session.delete(3) == "ran delete 3"
    assert session.destroy("tran1") == "ran destroy tran1"
    assert session.device_help("MOS") == "ran devhelp mos"
    assert session.save("v(out)") == "ran save v(out)"
    assert session.option("acct", reltol=1e-4) == "ran option acct\nran option reltol = 0.0001"
    assert session.set("noaskquit", filetype="ascii") == "ran set noaskquit\nran set filetype = ascii"
    assert session.unset("plotwinsize") == "ran unset plotwinsize"
    assert session.set_circuit("circ1") == "ran setcirc circ1"
    assert session.status() == "ran status"
    assert session.step() == "ran step"
    assert session.step(5) == "ran step 5"
    assert session.stop("v(out) > 1", after=10) == "ran stop after 10 when v(out) > 1"
    assert session.trace("v(out)", "i(v1)") == "ran trace v(out) i(v1)"
    assert session.where() == "ran where"
    assert session.listing() == "ran listing"
    assert session.quit() == "ran set noaskquit\nran quit"

    assert session.commands == [
        "delete 3",
        "destroy tran1",
        "devhelp mos",
        "save v(out)",
        "option acct",
        "option reltol = 0.0001",
        "set noaskquit",
        "set filetype = ascii",
        "unset plotwinsize",
        "setcirc circ1",
        "status",
        "step",
        "step 5",
        "stop after 10 when v(out) > 1",
        "trace v(out) i(v1)",
        "where",
        "listing",
        "set noaskquit",
        "quit",
    ]


def test_shared_session_command_wrappers_validate_command_parts():
    class CommandRecorder(NgspiceSharedSession):
        def __init__(self):
            pass

        def command(self, command):
            return command

    session = CommandRecorder()

    with pytest.raises(ValueError, match="cannot contain control characters"):
        session.save("v(out); quit")
    with pytest.raises(ValueError, match="cannot contain control characters"):
        session.option("acct\nquit")
    with pytest.raises(ValueError, match="cannot contain control characters"):
        session.option(**{"bad;name": 1})
    with pytest.raises(ValueError, match="cannot contain control characters"):
        session.set("noaskquit\nquit")
    with pytest.raises(ValueError, match="cannot contain control characters"):
        session.set(**{"bad;name": "value"})
    with pytest.raises(ValueError, match="number_of_steps must be positive"):
        session.step(0)
    with pytest.raises(ValueError, match="trace requires at least one vector"):
        session.trace()
    with pytest.raises(ValueError, match="unset requires at least one variable"):
        session.unset()


def test_shared_session_parses_show_and_resource_usage_output():
    class OutputSession(NgspiceSharedSession):
        def __init__(self):
            self.commands = []
            self.responses = {
                "show m1": "Device: M1\n gm = 0.001\n count = 4\n enabled = true\n",
                "showmod nch": "Model: nch\n version = 4\n type = nmos\n",
                "rusage elapsed temp": "elapsed: 1.5\ntemp = 27\naccepted = 12\nstable = yes\n",
                "rusage everything": "space = 1024\n",
            }

        def command(self, command):
            self.commands.append(command)
            return self.responses[command]

    session = OutputSession()

    assert session.show("M1") == {
        "description": "Device: M1",
        "Device": "M1",
        "gm": 0.001,
        "count": 4,
        "enabled": True,
    }
    assert session.showmod("NCH") == {
        "description": "Model: nch",
        "Model": "nch",
        "version": 4,
        "type": "nmos",
    }
    assert session.resource_usage("elapsed", "temp") == {
        "description": "elapsed: 1.5",
        "elapsed": 1.5,
        "temp": 27,
        "accepted": 12,
        "stable": True,
    }
    assert session.resource_usage() == {
        "description": "space = 1024",
        "space": 1024,
    }
    assert not hasattr(session, "ressource_usage")


def test_shared_session_sanity_when_libngspice_is_available():
    if not NgspiceSharedSession.available():
        pytest.skip("libngspice shared library is not available")

    with NgspiceSharedSession() as session:
        session.load_circuit(_rc_circuit())
        session.command("op")

        vectors = session.vectors()

    assert "in" in vectors
    assert "out" in vectors
    np.testing.assert_allclose(vectors["out"], np.array([0.5]), rtol=1e-6, atol=1e-9)


def test_shared_runner_sanity_when_libngspice_is_available():
    if not NgspiceSharedRunner.available():
        pytest.skip("libngspice shared library is not available")

    task = SimTask(
        circuit=_rc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        simulator="ngspice-shared",
        output_names=["out"],
    )

    result = LocalExecutor(max_workers=1, backend="ngspice-shared").submit(task).result()

    assert isinstance(result, SimResult)
    assert result.status == "ok"
    assert result.metadata["simulator"] == "ngspice-shared"
    np.testing.assert_allclose(result.waveforms["out"], np.array([0.0, 0.25, 0.5]), rtol=1e-6, atol=1e-9)
