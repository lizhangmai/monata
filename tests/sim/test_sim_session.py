import numpy as np
import pytest

from monata.netlist import Circuit, render_ngspice
from monata.netlist.ir import NetlistError
from monata.sim.analysis_spec import (
    ACSpec,
    DCSweep,
    OPSpec,
    TranSpec,
)
from monata.sim.core import SimulationSession
from monata.sim.task import SimTask
from support.executors import ImmediateExecutor, RecordingExecutor
from support.results import sim_result


def test_simulation_session_builds_directives_and_tasks():
    circuit = Circuit("session")
    session = (
        SimulationSession(circuit, simulator="ngspice-subprocess", output_names=["out", "out", "in"])
        .temperature(75, nominal=27)
        .options("acct", reltol="1e-4")
        .initial_condition(out=0)
        .node_set(in_=0)
        .save("v(out)")
        .save_internal_parameters("@m1[id]")
        .save_currents()
        .measure("tran", "tphl", "TRIG v(in) VAL=0.5 RISE=1", "TARG v(out) VAL=0.5 FALL=1")
        .raw_directive(".control")
        .raw_block(["run", "quit", ".endc"])
    )

    task = session.tran(1e-6, step=1e-9, metadata={"case": "nominal"})

    assert isinstance(task.analysis_spec, TranSpec)
    assert task.output_names == ("out", "in")
    assert task.metadata == {"case": "nominal"}
    assert render_ngspice(circuit) == (
        "session\n"
        ".options temp=75 tnom=27\n"
        ".options acct reltol=1e-4\n"
        ".ic v(out)=0\n"
        ".nodeset v(in)=0\n"
        ".save v(out)\n"
        ".save @m1[id] all\n"
        ".options savecurrents\n"
        ".measure tran tphl TRIG v(in) VAL=0.5 RISE=1 TARG v(out) VAL=0.5 FALL=1\n"
        ".control\n"
        "run\n"
        "quit\n"
        ".endc\n"
        ".end\n"
    )
    assert session.str_options() == (
        ".options temp=75 tnom=27\n"
        ".options acct reltol=1e-4\n"
        ".options savecurrents\n"
    )
    assert session.str_netlist() == render_ngspice(circuit)
    assert session.to_spice() == render_ngspice(circuit)
    assert str(session) == render_ngspice(circuit)
    assert not hasattr(session, "str")

    with pytest.raises(TypeError):
        session.str_options(unit=False)  # type: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        session.str_netlist(simulator=object())  # type: ignore[reportCallIssue]


def test_simulation_session_options_share_circuit_flag_validation():
    session = SimulationSession(Circuit("session"))

    with pytest.raises(NetlistError, match="option flag conflicts"):
        session.options("acct", acct=False)


def test_simulation_session_does_not_expose_ngspice_simulation_renderer():
    session = SimulationSession(Circuit("extended"))

    assert not hasattr(session, "str_simulation")


def test_simulation_session_can_disable_save_currents_option():
    circuit = Circuit("save currents")
    session = SimulationSession(circuit)

    session.save_currents().save_currents(False)
    session.options(savecurrents=True, reltol="1e-4").save_currents(False)

    assert render_ngspice(circuit) == (
        "save currents\n"
        ".options reltol=1e-4\n"
        ".end\n"
    )


def test_simulation_session_accepts_constructor_temperature_options():
    circuit = Circuit("init temp")

    session = SimulationSession(circuit, temperature=75, nominal_temperature=27)

    assert session.circuit is circuit
    assert render_ngspice(circuit) == (
        "init temp\n"
        ".options temp=75 tnom=27\n"
        ".end\n"
    )


def test_simulation_session_analysis_helpers_create_expected_specs():
    session = SimulationSession(Circuit("helpers"))

    assert isinstance(session.op().analysis_spec, OPSpec)
    assert isinstance(session.ac(start=1, stop=1e6, points=10).analysis_spec, ACSpec)
    assert session.tran(1e-6, max_step=1e-8).analysis_spec.max_step == 1e-8
    secondary = DCSweep("VSS", 0, 1.2, 0.1)
    assert session.dc("VDD", 0, 1.2, 0.1, secondary=secondary).analysis_spec.secondary is secondary
    noise = session.noise(
        "out",
        input_source="vin",
        start=1,
        stop=1e6,
        points=10,
        reference_node="ref",
        variation="lin",
        points_per_summary=2,
    )
    assert noise.analysis_spec.input_source == "vin"
    assert noise.analysis_spec.reference_node == "ref"
    assert noise.analysis_spec.variation == "lin"
    assert noise.analysis_spec.points_per_summary == 2
    assert session.sensitivity("v(out)").analysis_spec.output == "v(out)"
    assert not hasattr(session, "dc_sensitivity")
    ac_sens = session.ac_sensitivity("v(out)", start=10, stop=1e6, points=20, variation="oct")
    assert ac_sens.analysis_spec.output == "v(out)"
    assert ac_sens.analysis_spec.start == 10
    assert ac_sens.analysis_spec.stop == 1e6
    assert ac_sens.analysis_spec.points == 20
    assert ac_sens.analysis_spec.variation == "oct"
    pz = session.pole_zero("in", "0", "out", "0", transfer="cur", mode="zer")
    assert pz.analysis_spec.transfer == "cur"
    assert pz.analysis_spec.mode == "zer"
    distortion = session.distortion(start=1, stop=1e6, points=10, variation="lin", f2overf1=0.9)
    assert distortion.analysis_spec.points == 10
    assert distortion.analysis_spec.variation == "lin"
    assert distortion.analysis_spec.f2overf1 == 0.9
    assert session.transfer_function("v(out)", "vin").analysis_spec.input_source == "vin"
    fourier = session.fourier(1e6, "v(out)", 1e-6, step=1e-9, start=1e-9)
    assert fourier.analysis_spec.output == "v(out)"
    assert fourier.analysis_spec.step == 1e-9
    assert fourier.analysis_spec.start == 1e-9


def test_simulation_session_uses_monata_analysis_method_names():
    session = SimulationSession(Circuit("session method names"), metadata={"suite": "methods"})

    op = session.op(metadata={"case": "op"})
    tran = session.tran(1e-6, step=1e-9, start=1e-8, max_step=1e-7, uic=True)
    pz = session.pole_zero("in", "0", "out", "0", transfer="cur", mode="pol")
    tf = session.transfer_function("v(out)", "vin")
    ac = session.ac(start=10, stop=1e6, points=20, variation="oct")
    ac_sens = session.ac_sensitivity(
        "v(out)",
        start=100,
        stop=1e5,
        points=15,
        variation="lin",
    )
    distortion = session.distortion(
        start=1e3,
        stop=1e7,
        points=12,
        variation="dec",
        f2overf1=0.9,
    )
    noise = session.noise(
        "out",
        input_source="vin",
        reference_node="ref",
        start=1,
        stop=1e6,
        points=8,
        variation="oct",
        points_per_summary=2,
    )

    assert isinstance(op.analysis_spec, OPSpec)
    assert op.metadata == {"suite": "methods", "case": "op"}
    assert not hasattr(session, "operating_point")
    assert isinstance(tran.analysis_spec, TranSpec)
    assert tran.analysis_spec.step == 1e-9
    assert tran.analysis_spec.stop == 1e-6
    assert tran.analysis_spec.start == 1e-8
    assert tran.analysis_spec.max_step == 1e-7
    assert tran.analysis_spec.uic is True
    assert not hasattr(session, "transient")
    assert pz.analysis_spec.transfer == "cur"
    assert pz.analysis_spec.mode == "pol"
    assert not hasattr(session, "polezero")
    assert tf.analysis_spec.output == "v(out)"
    assert tf.analysis_spec.input_source == "vin"
    assert not hasattr(session, "tf")
    assert ac.analysis_spec.start == 10
    assert ac.analysis_spec.stop == 1e6
    assert ac.analysis_spec.points == 20
    assert ac.analysis_spec.variation == "oct"
    assert ac_sens.analysis_spec.start == 100
    assert ac_sens.analysis_spec.stop == 1e5
    assert ac_sens.analysis_spec.points == 15
    assert ac_sens.analysis_spec.variation == "lin"
    assert distortion.analysis_spec.start == 1e3
    assert distortion.analysis_spec.stop == 1e7
    assert distortion.analysis_spec.points == 12
    assert distortion.analysis_spec.f2overf1 == 0.9
    assert noise.analysis_spec.output_node == "out"
    assert noise.analysis_spec.reference_node == "ref"
    assert noise.analysis_spec.input_source == "vin"
    assert noise.analysis_spec.start == 1
    assert noise.analysis_spec.stop == 1e6
    assert noise.analysis_spec.points == 8
    assert noise.analysis_spec.variation == "oct"
    assert noise.analysis_spec.points_per_summary == 2

    with pytest.raises(TypeError, match="ac requires start, stop, and points"):
        session.ac(start_frequency=10, stop_frequency=1e6, number_of_points=20)
    with pytest.raises(TypeError, match="noise requires input_source"):
        session.noise("out", src="vin", start=1, stop=1e6, points=8)


def test_circuit_simulator_creates_backend_neutral_session():
    circuit = Circuit("session facade")

    assert not hasattr(Circuit, "simulation")

    session = circuit.simulator(
        simulator="ngspice-shared",
        output_names=["out", "out", "in"],
        metadata={"suite": "sanity"},
        backend_options={"rawfile_format": "binary"},
        artifacts={"directory": "artifacts", "overwrite": True},
        snapshot_tasks=False,
        timeout=None,
        temperature=125,
        nominal_temperature=27,
    )

    assert isinstance(session, SimulationSession)
    assert session.circuit is circuit
    assert session.simulator == "ngspice-shared"
    assert session.output_names == ("out", "in")
    assert session.metadata == {"suite": "sanity"}
    assert session.backend_options == {"rawfile_format": "binary"}
    assert str(session.artifacts.directory) == "artifacts"
    assert session.artifacts.overwrite is True
    assert session.snapshot_tasks is False
    assert session.timeout is None
    assert render_ngspice(circuit) == (
        "session facade\n"
        ".options temp=125 tnom=27\n"
        ".end\n"
    )


def test_simulation_session_applies_and_overrides_task_execution_options(tmp_path):
    circuit = Circuit("task options")
    session = SimulationSession(
        circuit,
        backend_options={"rawfile_format": "ascii"},
        artifacts=tmp_path / "session-artifacts",
    )

    inherited = session.tran(1e-6)
    overridden = session.tran(
        1e-6,
        backend_options={"rawfile_format": "binary"},
        artifacts={"directory": tmp_path / "task-artifacts", "overwrite": True},
    )

    assert inherited.backend_options == {"rawfile_format": "ascii"}
    assert inherited.artifacts.directory == tmp_path / "session-artifacts"
    assert inherited.artifacts.overwrite is False
    assert overridden.backend_options == {"rawfile_format": "binary"}
    assert overridden.artifacts.directory == tmp_path / "task-artifacts"
    assert overridden.artifacts.overwrite is True


def test_simulation_session_tasks_snapshot_circuit_by_default():
    circuit = Circuit("snapshots")
    session = SimulationSession(circuit)
    first = session.tran(1e-6)

    session.options(reltol="1e-4")
    second = session.tran(1e-6)

    assert first.circuit is not circuit
    assert second.circuit is not circuit
    assert render_ngspice(first.circuit) == "snapshots\n.end\n"
    assert render_ngspice(second.circuit) == "snapshots\n.options reltol=1e-4\n.end\n"


def test_simulation_session_can_share_mutable_circuit_explicitly():
    circuit = Circuit("shared")
    session = SimulationSession(circuit, snapshot_tasks=False)

    shared = session.tran(1e-6)
    snapshot = session.tran(1e-6, snapshot=True)

    assert shared.circuit is circuit
    assert snapshot.circuit is not circuit


def test_simulation_session_run_uses_supplied_executor():
    circuit = Circuit("run")
    session = SimulationSession(circuit)
    result = sim_result(
        waveforms={"out": np.array([1.0])},
        sweep_var=np.array([0.0]),
        corner=None,
    )
    executor = ImmediateExecutor(result)

    assert session.run(TranSpec(stop=1e-6), executor=executor) is result
    assert executor.task is not None
    assert executor.task.circuit is not circuit
    assert render_ngspice(executor.task.circuit) == render_ngspice(circuit)
    assert isinstance(executor.task.analysis_spec, TranSpec)


def test_simulation_session_can_queue_and_clear_analysis_tasks():
    circuit = Circuit("queue")
    session = SimulationSession(circuit, output_names=["out"], metadata={"suite": "nominal"})

    tran_task = session.queue(TranSpec(stop=1e-6), metadata={"case": "tran"})
    ac_task = session.queue(ACSpec(start=1, stop=1e6, points=10))
    op_task = session.op(metadata={"case": "op"})
    returned = session.queue(op_task)

    assert returned is op_task
    assert session.queued_tasks == (tran_task, ac_task, op_task)
    assert list(session.iter_queued_tasks()) == [tran_task, ac_task, op_task]
    assert not hasattr(session, "analysis_iter")
    assert tran_task.output_names == ("out",)
    assert tran_task.metadata == {"suite": "nominal", "case": "tran"}
    assert ac_task.metadata == {"suite": "nominal"}
    assert op_task not in (tran_task, ac_task)

    snapshot = session.queued_tasks
    session.clear_queue()

    assert snapshot == (tran_task, ac_task, op_task)
    assert session.queued_tasks == ()

    session.queue(OPSpec())
    assert not hasattr(session, "reset_analysis")
    assert session.clear_queue() is session
    assert session.queued_tasks == ()


def test_simulation_session_rejects_queued_task_with_extra_task_kwargs():
    session = SimulationSession(Circuit("queue_error"))
    task = session.op()

    try:
        session.queue(task, metadata={"case": "bad"})
    except ValueError as exc:
        assert "queued SimTask" in str(exc)
    else:
        raise AssertionError("expected queued SimTask keyword arguments to be rejected")
    assert session.queued_tasks == ()


def test_simulation_session_run_queue_uses_executor_map_and_preserves_order():
    session = SimulationSession(Circuit("run_queue"), output_names=["out"])
    tran_task = session.queue(TranSpec(stop=1e-6), metadata={"case": "tran"})
    ac_task = session.queue(ACSpec(start=1, stop=1e6, points=10), metadata={"case": "ac"})

    def queue_result(task: SimTask, index: int):
        return sim_result(
            waveforms={"out": np.array([float(index)])},
            sweep_var=np.array([0.0]),
            metadata={
                "case": task.metadata["case"],
                "index": index,
            },
        )

    executor = RecordingExecutor(queue_result)

    results = session.run_queue(executor=executor)

    assert executor.tasks == [tran_task, ac_task]
    assert [result.metadata["case"] for result in results] == ["tran", "ac"]
    assert [result.metadata["index"] for result in results] == [0, 1]
    assert session.queued_tasks == (tran_task, ac_task)

    cleared_results = session.run_queue(executor=executor, clear=True)

    assert [result.metadata["case"] for result in cleared_results] == ["tran", "ac"]
    assert session.queued_tasks == ()


def test_simulation_session_run_queue_returns_empty_for_empty_queue():
    session = SimulationSession(Circuit("empty_queue"))
    executor = RecordingExecutor()

    assert session.run_queue(executor=executor, clear=True) == []
    assert executor.called is False
