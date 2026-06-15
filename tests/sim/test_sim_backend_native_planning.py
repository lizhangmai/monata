from pathlib import Path

import pytest

from monata.corner import OperatingCorner
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
    device_parameter_vector,
    expression_vector,
)
from monata.sim.backends.ngspice import _control_block, _with_control_block
from monata.sim.backends.ngspice_plan import NgspiceTaskPlan, task_plan as _task_plan
from support.native_backend_cases import _ac_circuit, _dc_circuit, _tran_circuit
pytestmark = pytest.mark.native

def test_native_ngspice_core_analysis_command_emission():
    cases = [
        (TranSpec(stop=10, step=1, start=2, max_step=0.5, uic=True), "tran 1 10 2 0.5 uic", "tran"),
        (ACSpec(start=1, stop=1000, points=10, variation="oct"), "ac oct 10 1 1000", "ac"),
        (DCSpec("VDD", 0, 5, 1, secondary=DCSweep("temp", 25, 85, 10)), "dc VDD 0 5 1 temp 25 85 10", "dc"),
        (OPSpec(), "op", "op"),
    ]

    for spec, command, analysis_name in cases:
        task = SimTask(circuit=_dc_circuit(), analysis_spec=spec, output_names=["in"])
        plan = _task_plan(task)

        assert plan is not None
        assert plan.command == command
        assert plan.analysis_name == analysis_name


def test_native_ngspice_planner_keeps_explicit_non_ac_vectors_unwrapped():
    outputs = (
        "v(out)",
        branch_current_vector("V1"),
        device_parameter_vector("M1", "gm"),
        expression_vector("v(in)-v(out)"),
    )
    task = SimTask(
        circuit=_tran_circuit(),
        analysis_spec=TranSpec(step=1e-9, stop=5e-9),
        output_names=outputs,
    )

    plan = _task_plan(task)

    assert plan is not None
    assert plan.output_names == outputs
    assert plan.output_vectors == outputs
    assert [request["vector_kind"] for request in plan.metadata["output_requests"]] == [
        "node_voltage",
        "branch_current",
        "element_parameter",
        "expression",
    ]


def test_native_ngspice_planner_uses_raw_ac_vector_for_bare_node_magnitude():
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=ACSpec(start=1, stop=1e3, points=3),
        output_names=["out", "mag(v(out))"],
    )

    plan = _task_plan(task)

    assert plan is not None
    assert plan.output_vectors == ("v(out)", "v(out)")
    assert [request.transform for request in plan.output_requests] == ["mag", "mag"]
    assert plan.metadata["output_requests"][0]["output_name"] == "out"
    assert plan.metadata["output_requests"][0]["vector_kind"] == "node_voltage"
    assert plan.metadata["output_requests"][1]["output_name"] == "mag(v(out))"


def test_native_ngspice_tran_max_step_command_emission():
    task = SimTask(
        circuit=_tran_circuit(),
        analysis_spec=TranSpec(step=1e-9, stop=5e-9, start=1e-9, max_step=5e-10, uic=True),
        output_names=["out"],
    )

    plan = _task_plan(task)

    assert plan is not None
    assert plan.command == "tran 1e-09 5e-09 1e-09 5e-10 uic"
    assert plan.metadata["max_step"] == 5e-10


def test_native_ngspice_noise_extended_parameters_command_emission():
    task = SimTask(
        circuit=_ac_circuit(),
        analysis_spec=NoiseSpec(
            output_node="out",
            reference_node="in",
            input_source="I1",
            variation="lin",
            points=7,
            start=10,
            stop=1000,
            points_per_summary=2,
        ),
    )

    plan = _task_plan(task)

    assert plan is not None
    assert plan.command == "noise v(out,in) I1 lin 7 10 1000 2"
    assert plan.metadata["noise_reference_node"] == "in"
    assert plan.metadata["variation"] == "lin"
    assert plan.metadata["points_per_summary"] == 2


def test_p3_analysis_command_emission():
    cases = [
        (SensitivitySpec(output="v(out)"), "sens v(out)", "sens"),
        (SensitivitySpec(output="v(out)", start=10, stop=1000, points=7, variation="oct"), "sens v(out) ac oct 7 10 1000", "sens"),
        (PoleZeroSpec(input_pos="in", input_neg="0", output_pos="out", output_neg="0"), "pz in 0 out 0 vol pz", "pz"),
        (PoleZeroSpec(input_pos="in", input_neg="0", output_pos="out", output_neg="0", transfer="cur", mode="zer"), "pz in 0 out 0 cur zer", "pz"),
        (DistortionSpec(start=1, stop=1e3, points=3), "disto dec 3 1 1000.0", "disto"),
        (DistortionSpec(start=1, stop=1e3, points=3, variation="lin", f2overf1=0.9), "disto lin 3 1 1000.0 0.9", "disto"),
        (TransferFunctionSpec(output="v(out)", input_source="V1"), "tf v(out) V1", "tf"),
        (FourierSpec(frequency=1000, output="v(out)", stop=0.002, step=1e-5), "fourier 1000 v(out)", "four"),
        (FourierSpec(frequency=1000, output="v(out)", stop=0.002, step=1e-5, start=1e-5), "tran 1e-05 0.002 1e-05\nfourier 1000 v(out)", "four"),
    ]

    for spec, command_text, analysis_name in cases:
        plan = _task_plan(SimTask(circuit=_ac_circuit(), analysis_spec=spec))
        assert plan is not None
        assert command_text in plan.command
        assert plan.analysis_name == analysis_name
        assert plan.metadata["analysis"] == analysis_name


def test_ngspice_task_plan_default_extraction_matches_rawfile_first_runner():
    plan = NgspiceTaskPlan(
        analysis_name="dc",
        output_names=("in",),
        output_vectors=("v(in)",),
        output_requests=(),
        command="dc V1 0 1 0.5",
        osdi_paths=(),
        metadata={},
    )

    assert plan.extraction == "rawfile"


def test_param_overrides_render_as_ngspice_params():
    circuit = _dc_circuit()
    circuit.param("rload", "1k")
    task = SimTask(
        circuit=circuit,
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        param_overrides={"rload": "2k"},
    )

    plan = _task_plan(task)
    assert plan is not None
    deck = _with_control_block("title\n.param rload=1k\n.end\n", plan, task, Path("result.dat"))

    assert ".param rload=2k" in deck
    assert deck.index(".param rload=1k") < deck.index(".param rload=2k")


def test_structured_param_override_mutates_rendered_circuit():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        param_overrides={"R1.R": "2k"},
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok"
    assert result.metadata["structured_mutations"] == [
        {"target": "R1.R", "kind": "structured"}
    ]


def test_osdi_paths_render_before_analysis(tmp_path):
    osdi = tmp_path / "model.osdi"
    osdi.write_text("")
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        osdi_paths=[osdi],
    )

    plan = _task_plan(task)
    assert plan is not None
    control = _control_block(plan, tmp_path / "result.dat")

    quoted_osdi = f"'{osdi}'"
    assert f"pre_osdi {quoted_osdi}" in control
    assert control.index(f"pre_osdi {quoted_osdi}") < control.index("dc V1 0 1 0.5")
    assert plan.metadata["osdi_paths"] == [str(osdi)]


def test_control_block_emits_rawfile_commands_in_order(tmp_path):
    output_path = tmp_path / "results dir" / "result raw.raw"
    osdi_path = tmp_path / "model dir" / "model file.osdi"
    output_path.parent.mkdir()
    osdi_path.parent.mkdir()
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

    control = _control_block(plan, output_path)

    assert control == [
        ".control",
        "set wr_singlescale",
        f"pre_osdi '{osdi_path}'",
        "dc V1 0 1 0.5",
        "set filetype=binary",
        f"write '{output_path}' v(out)",
        f"wrdata '{output_path.with_suffix('.dat')}' v(out)",
        "quit",
        ".endc",
    ]


def test_corner_temperature_and_model_file_render_into_deck(tmp_path):
    model_file = tmp_path / "tt.mod"
    model_file.write_text(".model dummy nmos\n")
    corner = OperatingCorner("tt_25C", 25, voltages={}, process="tt", model_file=str(model_file))
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        corner=corner,
    )

    plan = _task_plan(task)
    assert plan is not None
    deck = _with_control_block("title\n.end\n", plan, task, tmp_path / "result.dat")

    assert ".temp 25" in deck
    assert f'.include "{model_file}"' in deck


def test_canonical_corner_lib_section_renders_once_when_not_projected(tmp_path):
    model_file = tmp_path / "ptm.mod"
    model_file.write_text(".lib tt\n.endl tt\n")
    corner = OperatingCorner(
        "tt_25C",
        25,
        voltages={},
        process="tt",
        model_file=str(model_file),
        techlib="PTM",
        model_deck="ptm",
        section="tt",
    )
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        corner=corner,
    )
    plan = _task_plan(task)
    assert plan is not None

    deck = _with_control_block("title\n.end\n", plan, task, tmp_path / "result.dat")
    title_contains_path_deck = _with_control_block(
        f"title mentions {model_file}\n.end\n",
        plan,
        task,
        tmp_path / "result.dat",
    )
    projected_deck = _with_control_block(
        f"title\n.lib {model_file} tt\n.end\n",
        plan,
        task,
        tmp_path / "result.dat",
    )

    assert f'.lib "{model_file}" tt' in deck
    assert f'.lib "{model_file}" tt' in title_contains_path_deck
    assert projected_deck.count(str(model_file)) == 1


def test_corner_lib_section_allows_strict_safe_token(tmp_path):
    model_file = tmp_path / "ptm.mod"
    model_file.write_text(".lib ss_125c\n.endl ss_125c\n")
    corner = OperatingCorner(
        "ss_125c",
        125,
        voltages={},
        process="ss",
        model_file=str(model_file),
        section="ss_125c",
    )
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        corner=corner,
    )
    plan = _task_plan(task)
    assert plan is not None

    deck = _with_control_block("title\n.end\n", plan, task, tmp_path / "result.dat")

    assert f'.lib "{model_file}" ss_125c' in deck


def test_corner_voltage_override_fails_explicitly():
    task = SimTask(
        circuit=_dc_circuit(),
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["in"],
        corner=OperatingCorner("hot", 125, voltages={"vdd": 0.9}),
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "failed"
    assert result.metadata["reason"] == "unsupported_param_overrides"
    assert "mutation target not found" in (result.error_message or "")
