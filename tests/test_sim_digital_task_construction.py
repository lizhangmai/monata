from pathlib import Path

import numpy as np
import pytest

from monata.corner import OperatingCorner
from monata.netlist import Circuit, render_ngspice
from monata.sim.core import SimResult, SimTask, TranSpec
from monata.sim.digital_claims import (
    DigitalTransientObservation,
)
from monata.sim.digital_plan import digital_stimulus_metadata
from monata.sim.digital_projection import (
    PdkDeviceProjection,
    PdkModelProjectionLibrary,
)
from monata.sim.digital_table import (
    DigitalTruthTable,
)
from monata.sim.export import sim_result_from_dict, sim_result_to_dict
from monata.sim.analysis_spec import OPSpec
from monata.sim.backends.ngspice import _control_block
from monata.sim.backends.ngspice_stdout import parse_op_print
from monata.sim.backends.ngspice_plan import task_plan
from support.digital_cases import (
    AND2_EXPECTED_TABLE,
    And2,
    BarePdkInverter,
    OBSERVED_CLAIM,
    PdkInverter,
    RecordingProjectionLibrary,
    _digital_task_metadata,
    _expected_and,
    _has_pwl_point,
    _pwl_points,
    _sequence_result_for_task,
)
pytestmark = pytest.mark.slow

def test_digital_sequence_skew_step_offsets_input_sources_and_schedule():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        period=2e-9,
        transition=1e-10,
        skew_step=2.5e-10,
    )

    task = table.propagation_delay_task()
    text = render_ngspice(task.circuit)
    a_points = _pwl_points(task.circuit.element("a").value.values)
    b_points = _pwl_points(task.circuit.element("b").value.values)
    schedule = table._plan.scheduled_single_bit_sequence_arcs(
        table._plan.single_bit_arcs(),
        initial_settle=0.0,
        cycles_per_vector=table.cycles_per_vector,
    )
    first_b_arc = next(arc for arc in schedule if arc.input_index == 1)
    first_a_arc = next(arc for arc in schedule if arc.input_index == 0)
    payload = _digital_task_metadata(task)
    stimulus = payload["stimulus"]

    assert stimulus["skew_step"] == 2.5e-10
    assert first_a_arc.trigger_start == pytest.approx(first_a_arc.start + 2 * table.period)
    assert first_b_arc.trigger_start == pytest.approx(first_b_arc.start + 2 * table.period + 2.5e-10)
    assert "Vb b 0 PWL(" in text
    assert "Va a 0 PWL(" in text
    assert _has_pwl_point(a_points, first_a_arc.trigger_start, first_a_arc.from_inputs[0] * table.vdd)
    assert _has_pwl_point(a_points, first_a_arc.trigger_end, first_a_arc.to_inputs[0] * table.vdd)
    assert _has_pwl_point(b_points, first_b_arc.trigger_start, first_b_arc.from_inputs[1] * table.vdd)
    assert _has_pwl_point(b_points, first_b_arc.trigger_end, first_b_arc.to_inputs[1] * table.vdd)


def test_digital_stimulus_metadata_decoder_rejects_malformed_fields():
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=np.array([0.0, 1.0]),
        corner=None,
        metadata={
            "task_metadata": {
                "monata": {
                    "digital_task_v1": {
                        "schema": "monata.sim.digital-task.v1",
                        "stimulus": {
                            "kind": "digital_single_bit_arc_sequence",
                            "arcs": "bad",
                        },
                    }
                }
            }
        },
    )

    with pytest.raises(RuntimeError, match="arcs"):
        digital_stimulus_metadata(result)


def test_digital_sequence_metadata_rejects_inconsistent_slot_duration():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        period=2e-9,
        transition=1e-10,
    )
    task = table.propagation_delay_task()
    payload = _digital_task_metadata(task)
    stimulus = dict(payload["stimulus"])
    stimulus["slot_duration"] = float(stimulus["slot_duration"]) * 2
    digital_payload = {**payload, "stimulus": stimulus}
    monata_metadata = dict(task.metadata["monata"])
    monata_metadata["digital_task_v1"] = digital_payload
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=np.array([0.0, 1.0]),
        corner=None,
        metadata={"task_metadata": {**task.metadata, "monata": monata_metadata}},
    )

    with pytest.raises(RuntimeError, match="slot_duration"):
        table._plan.sequence_for_result(result)


def test_transient_tasks_build_digital_sequence():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        oracle="observed",
        step=5e-11,
        corner="ptm65",
    )

    tasks = table.transient_tasks(stop=5e-8, uic=True)
    text = render_ngspice(tasks[-1].circuit)
    plan = task_plan(tasks[-1])

    assert len(tasks) == 1
    assert all(isinstance(task.analysis_spec, TranSpec) for task in tasks)
    assert tasks[-1].corner is not None
    assert tasks[-1].corner.name == "ptm65"
    assert tasks[-1].metadata["claim"] == OBSERVED_CLAIM
    assert "stimulus" not in tasks[-1].metadata
    assert "measurements" not in tasks[-1].metadata
    payload = _digital_task_metadata(tasks[-1])
    assert payload["schema"] == "monata.sim.digital-task.v1"
    assert payload["digital_truth_table"]["task_kind"] == "digital-single-bit-arc-sequence"
    assert payload["measurements"] == ["truth_table"]
    assert payload["stimulus"]["kind"] == "digital_single_bit_arc_sequence"
    assert payload["stimulus"]["arc_coverage"] == "directed_single_bit_exhaustive"
    assert payload["stimulus"]["total_arcs"] == 8
    assert payload["stimulus"]["chunk_count"] == 1
    assert plan is not None
    assert plan.command.startswith("tran ")
    assert "Va a 0 PWL(" in text
    assert "Vb b 0 PWL(" in text
    assert "Xdut a b out vdd 0 and2" in text


def test_transient_observation_config_is_public_task_api():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        oracle="observed",
        step=5e-11,
    )
    observation = DigitalTransientObservation(stop=5e-8, uic=True)

    tasks = table.transient_observation_tasks(observation)
    override_tasks = table.transient_observation_tasks({"stop": "1e-8"}, uic=True)

    assert observation.as_dict() == {
        "stop": 5e-8,
        "uic": True,
        "cycles_per_vector": None,
        "slots_per_task": None,
    }
    assert len(tasks) == 1
    assert isinstance(tasks[-1].analysis_spec, TranSpec)
    payload = _digital_task_metadata(tasks[-1])
    assert payload["stimulus"]["kind"] == "digital_single_bit_arc_sequence"
    assert payload["stimulus"]["initial_settle"] == pytest.approx(5e-8)
    assert payload["stimulus"]["cycles_per_vector"] == 2
    assert isinstance(override_tasks[-1].analysis_spec, TranSpec)
    assert _digital_task_metadata(override_tasks[-1])["stimulus"]["initial_settle"] == pytest.approx(1e-8)
    with pytest.raises(ValueError, match="stop must be positive"):
        DigitalTransientObservation.resolve({"stop": 0})


def test_digital_sequence_uses_transient_step():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        oracle="observed",
        period=4e-9,
        step=5e-12,
        truth_table_step=2e-10,
        transition=1e-9,
    )

    truth_task = table.transient_tasks()[0]
    delay_task = table.propagation_delay_task()

    assert isinstance(truth_task.analysis_spec, TranSpec)
    assert _digital_task_metadata(truth_task)["stimulus"]["kind"] == "digital_single_bit_arc_sequence"
    truth_spec = truth_task.analysis_spec
    assert isinstance(truth_spec, TranSpec)
    assert truth_spec.step == pytest.approx(5e-12)
    delay_spec = delay_task.analysis_spec
    assert isinstance(delay_spec, TranSpec)
    assert delay_spec.step == pytest.approx(5e-12)


def test_run_transient_observation_uses_configured_tasks():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=_expected_and,
        threshold=0.5,
    )
    observation = DigitalTransientObservation(stop=2.5e-8, uic=True)

    class Future:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class RecordingExecutor:
        def __init__(self):
            self.tasks = []
            self.results = []

        def map(self, tasks):
            self.tasks = list(tasks)
            results = []
            for task in self.tasks:
                sim_result = _sequence_result_for_task(table, task)
                self.results.append(sim_result)
                results.append(Future(sim_result))
            return results

    executor = RecordingExecutor()

    result = table.run_transient_observation(executor, observation=observation)

    assert len(executor.tasks) == 1
    assert _digital_task_metadata(executor.tasks[0])["stimulus"]["kind"] == "digital_single_bit_arc_sequence"
    assert result.mode == "transient"
    assert result.sim_result is executor.results[0]
    assert result.sim_results == tuple(executor.results)
    assert [row.as_dict()["status"] for row in result] == ["PASS", "PASS", "PASS", "PASS"]


def test_digital_task_metadata_namespace_survives_result_export_round_trip():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        metadata={"owner": "unit-test"},
    )
    task = table.transient_tasks()[0]
    result = _sequence_result_for_task(table, task)

    array_store = {}
    payload = sim_result_to_dict(result, array_store=array_store)
    restored = sim_result_from_dict(payload, array_store=array_store)
    restored_stimulus = digital_stimulus_metadata(restored)

    assert task.metadata["owner"] == "unit-test"
    assert "stimulus" not in task.metadata
    assert _digital_task_metadata(restored)["measurements"] == ["truth_table"]
    assert restored_stimulus.kind == "digital_single_bit_arc_sequence"


def test_digital_task_metadata_rejects_unknown_schema():
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=np.array([0.0]),
        corner=None,
        metadata={
            "task_metadata": {
                "monata": {"digital_task_v1": {"schema": "monata.sim.digital-task.v999"}}
            }
        },
    )

    with pytest.raises(RuntimeError, match="unsupported digital task metadata schema"):
        digital_stimulus_metadata(result)


def test_digital_task_metadata_requires_result_task_metadata_envelope():
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=np.array([0.0]),
        corner=None,
        metadata={
            "monata": {
                "digital_task_v1": {
                    "schema": "monata.sim.digital-task.v1",
                    "stimulus": {"kind": "digital_single_bit_arc_sequence"},
                }
            }
        },
    )

    with pytest.raises(RuntimeError, match="must include task_metadata"):
        digital_stimulus_metadata(result)


def test_digital_task_metadata_requires_monata_namespace():
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=np.array([0.0]),
        corner=None,
        metadata={"task_metadata": {"stimulus": {"kind": "digital_single_bit_arc_sequence"}}},
    )

    with pytest.raises(RuntimeError, match=r"metadata\['monata'\]"):
        digital_stimulus_metadata(result)


def test_digital_task_metadata_rejects_missing_schema():
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=np.array([0.0]),
        corner=None,
        metadata={"task_metadata": {"monata": {"digital_task_v1": {"stimulus": {}}}}},
    )

    with pytest.raises(RuntimeError, match="unsupported digital task metadata schema"):
        digital_stimulus_metadata(result)


def test_digital_task_metadata_rejects_non_mapping_payload():
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=np.array([0.0]),
        corner=None,
        metadata={"task_metadata": {"monata": {"digital_task_v1": "bad"}}},
    )

    with pytest.raises(RuntimeError, match="payload must be a mapping"):
        digital_stimulus_metadata(result)


def test_projection_library_runs_after_task_circuits_are_assembled():
    library = RecordingProjectionLibrary()
    table = DigitalTruthTable(
        PdkInverter,
        inputs=("vin",),
        outputs=("out",),
        library=library,
        corner="ptm65",
        load_cap="1f",
    )

    vector_tasks = table.transient_tasks()

    assert len(library.calls) == 1
    assert all(isinstance(call["corner"], OperatingCorner) for call in library.calls)
    assert all(call["corner_name"] == "ptm65" for call in library.calls)
    assert all("pdk_inv" in call["subcircuit_names"] for call in library.calls)
    assert all(call["pdk_instance_counts"] == [1] for call in library.calls)
    for task in vector_tasks:
        assert task.corner is not None
        assert task.corner.name == "ptm65"
    assert all(".options projected=ptm65" in render_ngspice(task.circuit) for task in vector_tasks)


def test_pdk_model_projection_library_lowers_source_devices_to_models():
    library = PdkModelProjectionLibrary(
        lib="PTM_BULK",
        view="ngspice",
        devices={"nmos": PdkDeviceProjection(model="nmos", pins=("d", "g", "s", "b"))},
    )
    table = DigitalTruthTable(
        PdkInverter,
        inputs=("vin",),
        outputs=("out",),
        library=library,
        load_cap="1f",
    )

    text = render_ngspice(table.transient_tasks()[0].circuit)

    assert "Mmn out vin gnd gnd nmos w=1.2u l=65n" in text


def test_pdk_model_projection_library_applies_projection_params():
    library = PdkModelProjectionLibrary(
        lib="PTM_MG",
        view="ngspice",
        devices={
            "nfet": PdkDeviceProjection(
                model="nmos",
                pins=("d", "g", "s", "b"),
                params={"w": "1.2u", "l": "65n"},
            )
        },
    )
    table = DigitalTruthTable(
        BarePdkInverter,
        inputs=("vin",),
        outputs=("out",),
        library=library,
        load_cap="1f",
    )

    text = render_ngspice(table.transient_tasks()[0].circuit)

    assert "Mmn out vin gnd gnd nmos w=1.2u l=65n" in text
    assert "Mmn_explicit out2 vin gnd gnd nmos w=1.2u l=50n" in text


def test_pdk_model_projection_library_rejects_registry_or_corner_projection():
    library = PdkModelProjectionLibrary(
        lib="PTM_BULK",
        view="ngspice",
        devices={"nmos": PdkDeviceProjection(model="nmos", pins=("d", "g", "s", "b"))},
    )

    with pytest.raises(ValueError, match="does not support registry or corner"):
        library.project_pdk_instances(PdkInverter(), registry=object())

    with pytest.raises(ValueError, match="does not support registry or corner"):
        library.project_pdk_instances(PdkInverter(), corner="ptm65")


def test_op_control_prints_are_chunked_for_large_output_sets():
    task = SimTask(
        circuit=Circuit("chunked OP print"),
        analysis_spec=OPSpec(),
        output_names=tuple(f"out_{index}" for index in range(70)),
    )
    ngspice_plan = task_plan(task)
    assert ngspice_plan is not None

    lines = _control_block(ngspice_plan, Path("result.dat"))

    print_lines = [line for line in lines if line.startswith("print ")]
    assert len(print_lines) > 1
    assert all(len(line.split()) <= 65 for line in print_lines)


def test_parse_op_print_is_case_insensitive_and_single_pass():
    waveforms = parse_op_print(
        "v(vec_00_p0) = 1.000000e+00\nv(vec_00_p1) = 0.000000e+00\n",
        ["vec_00_P0", "vec_00_P1"],
    )

    np.testing.assert_allclose(waveforms["vec_00_P0"], np.array([1.0]))
    np.testing.assert_allclose(waveforms["vec_00_P1"], np.array([0.0]))
