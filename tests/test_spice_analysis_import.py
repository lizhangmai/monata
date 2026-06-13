from pathlib import Path

import pytest

from monata.netlist import render_ngspice
from monata.parser import (
    SpiceAnalysisMeasurement,
    SpiceAnalysisSweep,
    UnsupportedConstructError,
    inspect_spice_analysis,
    spice_to_sim_tasks,
)
from monata.sim.analysis_spec import (
    ACSpec,
    DCSpec,
    DistortionSpec,
    FourierSpec,
    NoiseSpec,
    PoleZeroSpec,
    SensitivitySpec,
    TranSpec,
    TransferFunctionSpec,
)


def test_inspect_spice_analysis_projects_tasks_and_outputs():
    plan = inspect_spice_analysis(
        """
analysis deck
R1 in out 1k
C1 out 0 1u
.save v(out) v(in)
.tran 1n 10n 2n uic
.ac dec 10 1 1meg
.print ac db(v(out)) phase(v(out))
.noise v(out) Vin dec 5 10 1meg
.end
"""
    )

    assert plan.supported is True
    assert plan.task_count == 3
    tran, ac, noise = plan.steps

    assert isinstance(tran.analysis_spec, TranSpec)
    assert tran.output_names == ("out", "in")
    assert tran.analysis_spec.step == pytest.approx(1e-9)
    assert tran.analysis_spec.stop == pytest.approx(10e-9)
    assert tran.analysis_spec.start == pytest.approx(2e-9)
    assert tran.analysis_spec.uic is True

    assert isinstance(ac.analysis_spec, ACSpec)
    assert ac.output_names == ("v(out)", "v(in)", "db(v(out))", "phase(v(out))")
    assert ac.analysis_spec.stop == pytest.approx(1e6)

    assert isinstance(noise.analysis_spec, NoiseSpec)
    assert noise.output_names == ()
    assert noise.analysis_spec.output_node == "out"


def test_spice_analysis_imports_extended_analysis_parameters():
    plan = inspect_spice_analysis(
        """
extended analysis deck
V1 in 0 DC 0 AC 1
V2 bias 0 DC 0
R1 in out 1k
R2 out bias 1k
.save v(out)
.dc V1 0 1 0.5 V2 0 1 0.5
.tran 1n 10n 2n 0.5n uic
.noise v(out,in) V1 lin 5 10 1meg 2
.end
"""
    )

    assert plan.supported is True
    assert plan.task_count == 3
    dc, tran, noise = plan.steps

    assert isinstance(dc.analysis_spec, DCSpec)
    assert dc.analysis_spec.secondary is not None
    assert dc.analysis_spec.secondary.source == "V2"
    assert dc.analysis_spec.secondary.step == pytest.approx(0.5)

    assert isinstance(tran.analysis_spec, TranSpec)
    assert tran.analysis_spec.start == pytest.approx(2e-9)
    assert tran.analysis_spec.max_step == pytest.approx(0.5e-9)
    assert tran.analysis_spec.uic is True

    assert isinstance(noise.analysis_spec, NoiseSpec)
    assert noise.analysis_spec.output_node == "out"
    assert noise.analysis_spec.reference_node == "in"
    assert noise.analysis_spec.variation == "lin"
    assert noise.analysis_spec.points_per_summary == 2


def test_spice_analysis_import_projects_advanced_analysis_specs_without_output_directives():
    plan = inspect_spice_analysis(
        """
advanced analysis deck
V1 in 0 DC 0 AC 1
R1 in out 1k
.sens v(out)
.sens v(out) ac dec 5 10 1meg
.pz in 0 out 0 vol pz
.tf v(out) V1
.disto dec 7 10 1meg 0.9
.end
"""
    )

    assert plan.supported is True
    assert plan.task_count == 5
    dc_sens, ac_sens, pole_zero, transfer, distortion = plan.steps

    assert dc_sens.name == "sens"
    assert isinstance(dc_sens.analysis_spec, SensitivitySpec)
    assert dc_sens.analysis_spec.output == "v(out)"
    assert dc_sens.analysis_spec.start is None
    assert ac_sens.name == "sens"
    assert isinstance(ac_sens.analysis_spec, SensitivitySpec)
    assert ac_sens.analysis_spec.variation == "dec"
    assert ac_sens.analysis_spec.points == 5
    assert ac_sens.analysis_spec.start == pytest.approx(10)
    assert ac_sens.analysis_spec.stop == pytest.approx(1e6)

    assert pole_zero.name == "pz"
    assert isinstance(pole_zero.analysis_spec, PoleZeroSpec)
    assert pole_zero.analysis_spec.input_pos == "in"
    assert pole_zero.analysis_spec.output_pos == "out"
    assert pole_zero.analysis_spec.transfer == "vol"
    assert pole_zero.analysis_spec.mode == "pz"

    assert transfer.name == "tf"
    assert isinstance(transfer.analysis_spec, TransferFunctionSpec)
    assert transfer.analysis_spec.output == "v(out)"
    assert transfer.analysis_spec.input_source == "V1"

    assert distortion.name == "disto"
    assert isinstance(distortion.analysis_spec, DistortionSpec)
    assert distortion.analysis_spec.variation == "dec"
    assert distortion.analysis_spec.points == 7
    assert distortion.analysis_spec.f2overf1 == pytest.approx(0.9)

    tasks = plan.to_tasks()
    assert [task.metadata["import_analysis"] for task in tasks] == ["sens", "sens", "pz", "tf", "disto"]
    assert all(task.output_names == () for task in tasks)
    rendered = render_ngspice(tasks[0].circuit)
    assert ".sens" not in rendered
    assert ".pz" not in rendered
    assert ".tf" not in rendered
    assert ".disto" not in rendered


def test_spice_analysis_import_accepts_distortion_alias():
    plan = inspect_spice_analysis(
        """
distortion alias deck
V1 in 0 DC 0 AC 1
R1 in out 1k
.distortion oct 4 10 1meg
.end
"""
    )

    assert plan.supported is True
    assert plan.task_count == 1
    step = plan.steps[0]
    assert step.name == "disto"
    assert isinstance(step.analysis_spec, DistortionSpec)
    assert step.analysis_spec.variation == "oct"
    assert step.analysis_spec.f2overf1 is None


def test_spice_to_sim_tasks_removes_raw_analysis_directives_from_base_circuit():
    tasks = spice_to_sim_tasks(
        """
task deck
V1 in 0 DC 1
R1 in out 1k
.save v(out)
.tran 1n 10n
.end
""",
        metadata={"case": "imported"},
    )

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task.analysis_spec, TranSpec)
    assert task.output_names == ("out",)
    assert task.metadata["case"] == "imported"
    assert task.metadata["import_analysis"] == "tran"
    rendered = render_ngspice(task.circuit)
    assert ".tran" not in rendered
    assert ".save v(out)" in rendered


def test_spice_analysis_import_binds_measurements_to_matching_tasks():
    plan = inspect_spice_analysis(
        """
measurement deck
V1 in 0 DC 1 AC 1
R1 in out 1k
.save v(out)
.tran 1n 10n
.measure tran tphl TRIG v(in) VAL=0.5 RISE=1 TARG v(out) VAL=0.5 FALL=1
.ac dec 10 1 1meg
.measure ac gain_1k FIND v(out) AT=1k
.end
"""
    )

    assert plan.supported is True
    assert plan.measurement_count == 2
    assert isinstance(plan.measurements[0], SpiceAnalysisMeasurement)
    assert plan.measurements_for("tran")[0].name == "tphl"
    assert plan.measurements_for(".ac")[0].expression == "FIND v(out) AT=1k"

    tran_task, ac_task = plan.to_tasks()
    assert tran_task.metadata["import_measurements"][0]["name"] == "tphl"
    assert ac_task.metadata["import_measurements"][0]["analysis"] == "ac"

    tran_netlist = render_ngspice(tran_task.circuit)
    ac_netlist = render_ngspice(ac_task.circuit)
    assert ".measure tran tphl" in tran_netlist
    assert ".measure ac gain_1k" not in tran_netlist
    assert ".measure ac gain_1k FIND v(out) AT=1k" in ac_netlist
    assert ".measure tran tphl" not in ac_netlist


def test_spice_analysis_import_binds_meas_alias_to_matching_task():
    plan = inspect_spice_analysis(
        """
meas alias deck
V1 in 0 DC 1
R1 in out 1k
.save v(out)
.tran 1n 10n
.meas tran vmax MAX v(out)
.end
"""
    )

    assert plan.supported is True
    assert plan.measurement_count == 1
    assert plan.measurements_for("tran")[0].name == "vmax"

    task = plan.to_tasks()[0]
    assert task.metadata["import_measurements"][0]["raw"] == ".meas tran vmax MAX v(out)"
    assert ".measure tran vmax MAX v(out)" in render_ngspice(task.circuit)


def test_spice_analysis_import_expands_step_param_list_to_tasks():
    plan = inspect_spice_analysis(
        """
step list deck
.param rload=1k
V1 in 0 DC 1
R1 in out {rload}
.save v(out)
.step param rload list 1k 2k 4k
.tran 1n 10n
.end
"""
    )

    assert plan.supported is True
    assert plan.sweep_count == 1
    assert plan.task_count == 3
    assert isinstance(plan.sweeps[0], SpiceAnalysisSweep)
    assert plan.sweeps[0].target == "rload"
    assert plan.sweeps[0].values == ("1k", "2k", "4k")

    tasks = plan.to_tasks(metadata={"case": "imported"})
    assert [task.param_overrides["rload"] for task in tasks] == ["1k", "2k", "4k"]
    assert tasks[0].metadata["case"] == "imported"
    assert tasks[0].metadata["import_sweep_overrides"] == {"rload": "1k"}
    assert tasks[2].metadata["import_sweeps"][0]["value"] == "4k"
    assert ".step" not in render_ngspice(tasks[0].circuit)


def test_spice_analysis_import_expands_step_param_linear_to_tasks():
    tasks = spice_to_sim_tasks(
        """
step linear deck
V1 in 0 DC 1
R1 in out {rload}
.save v(out)
.step param rload 1k 3k 1k
.tran 1n 10n
.end
"""
    )

    assert len(tasks) == 3
    assert [task.param_overrides["rload"] for task in tasks] == pytest.approx([1e3, 2e3, 3e3])


def test_spice_analysis_import_expands_step_param_decade_to_tasks():
    tasks = spice_to_sim_tasks(
        """
step decade deck
V1 in 0 DC 1
R1 in out {rload}
.save v(out)
.step dec param rload 1k 100k 1
.tran 1n 10n
.end
"""
    )

    assert len(tasks) == 3
    assert [task.param_overrides["rload"] for task in tasks] == pytest.approx([1e3, 1e4, 1e5])
    assert tasks[0].metadata["import_sweeps"][0]["mode"] == "dec"


def test_spice_analysis_import_expands_step_param_octave_to_tasks():
    tasks = spice_to_sim_tasks(
        """
step octave deck
V1 in 0 DC 1
R1 in out {rload}
.save v(out)
.step oct param rload 1k 8k 1
.tran 1n 10n
.end
"""
    )

    assert len(tasks) == 4
    assert [task.param_overrides["rload"] for task in tasks] == pytest.approx([1e3, 2e3, 4e3, 8e3])
    assert tasks[-1].metadata["import_sweeps"][0]["mode"] == "oct"


def test_spice_analysis_import_expands_step_temperature_to_corners():
    tasks = spice_to_sim_tasks(
        """
temperature step deck
V1 in 0 DC 1
R1 in out 1k
.save v(out)
.step temp -40 40 40
.tran 1n 10n
.end
""",
        corner={"name": "tt", "temperature": 27, "voltages": {"vdd": 1.0}},
    )

    assert len(tasks) == 3
    assert [task.corner.temperature for task in tasks] == pytest.approx([-40, 0, 40])
    assert [task.corner.name for task in tasks] == ["tt_m40C", "tt_0C", "tt_40C"]
    assert tasks[0].corner.voltages["vdd"] == pytest.approx(1.0)
    assert tasks[0].metadata["import_sweeps"][0]["kind"] == "temperature"
    assert tasks[2].metadata["import_sweeps"][0]["value"] == pytest.approx(40)
    assert tasks[0].param_overrides == {}


def test_spice_analysis_import_expands_step_temperature_octave_to_corners():
    tasks = spice_to_sim_tasks(
        """
temperature octave step deck
V1 in 0 DC 1
R1 in out 1k
.save v(out)
.step oct temp 10 40 1
.tran 1n 10n
.end
"""
    )

    assert len(tasks) == 3
    assert [task.corner.temperature for task in tasks] == pytest.approx([10, 20, 40])
    assert [task.corner.name for task in tasks] == ["10C", "20C", "40C"]
    assert tasks[0].metadata["import_sweeps"][0]["kind"] == "temperature"
    assert tasks[0].metadata["import_sweeps"][0]["mode"] == "oct"


def test_spice_analysis_import_rejects_non_positive_log_step_bounds():
    plan = inspect_spice_analysis(
        """
bad log step deck
V1 in 0 DC 1
R1 in out {rload}
.save v(out)
.step dec param rload -1 100 1
.tran 1n 10n
.end
"""
    )

    assert plan.supported is False
    assert "positive start and stop" in plan.issues[0].message


def test_spice_analysis_import_rejects_multiple_temperature_sweeps():
    plan = inspect_spice_analysis(
        """
bad temperature step deck
V1 in 0 DC 1
R1 in out 1k
.save v(out)
.step temp list -40 27
.step temperature 85 125 40
.tran 1n 10n
.end
"""
    )

    assert plan.supported is False
    assert plan.sweep_count == 2
    assert "at most one temperature sweep" in plan.issues[0].message
    with pytest.raises(UnsupportedConstructError, match="at most one temperature sweep"):
        plan.to_tasks()


def test_spice_analysis_import_migrates_task_control_block():
    plan = inspect_spice_analysis(
        """
control task deck
V1 in 0 DC 1
R1 in out 1k
.control
tran 1n 10n
plot v(out)
meas tran vmax MAX v(out)
run
quit
.endc
.end
"""
    )

    assert plan.supported is True
    assert plan.task_count == 1
    assert plan.steps[0].origin == "control"
    assert plan.steps[0].output_names == ("out",)
    assert plan.measurements[0].origin == "control"

    task = plan.to_tasks()[0]
    rendered = render_ngspice(task.circuit)
    assert task.metadata["import_origin"] == "control"
    assert task.metadata["import_measurements"][0]["origin"] == "control"
    assert ".control" not in rendered
    assert "\ntran 1n 10n\n" not in rendered
    assert ".measure tran vmax MAX v(out)" in rendered


def test_spice_analysis_import_preserves_control_order_for_fourier():
    plan = inspect_spice_analysis(
        """
control fourier deck
R1 in out 1k
.control
tran 10u 2m
plot v(out)
four 1k v(out)
.endc
.end
"""
    )

    assert plan.supported is True
    assert plan.task_count == 2
    assert isinstance(plan.steps[1].analysis_spec, FourierSpec)
    assert plan.steps[1].analysis_spec.stop == pytest.approx(2e-3)


def test_spice_analysis_import_rejects_control_blocks_with_stateful_commands():
    plan = inspect_spice_analysis(
        """
stateful control task
V1 in 0 DC 1
R1 in out 1k
.control
set noaskquit
tran 1n 10n
plot v(out)
.endc
.end
"""
    )

    assert plan.supported is False
    assert plan.task_count == 0
    assert ".control block analysis import cannot migrate" in plan.issues[0].message
    with pytest.raises(UnsupportedConstructError, match="control block analysis import"):
        plan.to_tasks()


def test_spice_analysis_import_reports_malformed_measurements():
    plan = inspect_spice_analysis(
        """
bad measurement deck
R1 in out 1k
.save v(out)
.tran 1n 10n
.measure tran only_name
.end
"""
    )

    assert plan.supported is False
    assert plan.measurement_count == 0
    assert ".measure import requires analysis" in plan.issues[0].message
    with pytest.raises(UnsupportedConstructError, match="measure import"):
        plan.to_tasks()


def test_spice_analysis_import_reports_unrepresented_semantics():
    plan = inspect_spice_analysis(
        """
unsupported analysis
R1 in out 1k
.save v(out)
.dc V1 0 1 0.1 V2
.tran 1n
.noise out Vin lin 5 10 1meg
.end
"""
    )

    assert plan.supported is False
    assert plan.unsupported_count == 3
    messages = [issue.message for issue in plan.issues]
    assert any(".dc import expects" in message for message in messages)
    assert any(".tran import expects" in message for message in messages)
    assert any(".noise output must be a voltage expression" in message for message in messages)
    with pytest.raises(UnsupportedConstructError, match=".dc import expects"):
        plan.to_tasks()


def test_spice_analysis_import_allows_rawfile_tasks_without_explicit_outputs():
    plan = inspect_spice_analysis(
        """
implicit outputs
R1 in out 1k
.dc V1 0 1 0.5
.tran 1n 10n
.ac dec 3 1 1k
.op
.end
"""
    )

    assert plan.supported is True
    assert [step.action for step in plan.steps] == ["task", "task", "task", "task"]
    assert [step.name for step in plan.steps] == ["dc", "tran", "ac", "op"]
    assert all(step.output_names == () for step in plan.steps)
    tasks = plan.to_tasks()
    assert len(tasks) == 4
    assert all(task.output_names == () for task in tasks)


def test_spice_analysis_import_pairs_fourier_with_previous_tran():
    plan = inspect_spice_analysis(
        """
fourier deck
R1 in out 1k
.save v(out)
.tran 10u 2m
.four 1k v(out)
.end
"""
    )

    assert plan.supported is True
    assert isinstance(plan.steps[1].analysis_spec, FourierSpec)
    assert plan.steps[1].analysis_spec.frequency == pytest.approx(1e3)
    assert plan.steps[1].analysis_spec.stop == pytest.approx(2e-3)
    assert plan.steps[1].analysis_spec.step == pytest.approx(10e-6)


def test_spice_analysis_import_rejects_fourier_without_tran():
    plan = inspect_spice_analysis(
        """
bad fourier
R1 in out 1k
.four 1k v(out)
.end
"""
    )

    assert plan.supported is False
    assert "requires an earlier supported .tran" in plan.issues[0].message


def test_spice_analysis_import_reads_only_explicit_path_objects(tmp_path):
    deck_path = tmp_path / "analysis.cir"
    deck_path.write_text("file analysis\nR1 in out 1k\n.save v(out)\n.tran 1n 10n\n.end\n")

    from_path = inspect_spice_analysis(deck_path)
    inline = inspect_spice_analysis(str(deck_path))

    assert from_path.path == str(deck_path)
    assert from_path.task_count == 1
    assert inline.path is None
    assert inline.task_count == 0


def test_spice_analysis_import_path_accepts_path_subclasses(tmp_path):
    class CustomPath(Path):
        _flavour = type(tmp_path)._flavour

    deck_path = CustomPath(tmp_path / "custom.cir")
    deck_path.write_text("custom analysis\nR1 in out 1k\n.save v(out)\n.tran 1n 10n\n.end\n")

    assert inspect_spice_analysis(deck_path).task_count == 1
