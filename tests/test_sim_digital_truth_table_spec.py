
import numpy as np
import pytest

from monata.netlist import render_ngspice
from monata.sim.core import SimResult, TranSpec
from monata.sim.digital_claims import (
    DigitalTransientObservation,
)
from monata.sim.digital_table import (
    DigitalTruthTable,
    bits_to_text,
    resolve_digital_truth_table_mode,
)
from support.digital_cases import (
    AND2_EXPECTED_TABLE,
    And2,
    HelperCell,
    LOGIC_GATE_CASES,
    LogicGateCase,
    OBSERVED_CLAIM,
    _digital_task_metadata,
    _expected_and,
    _sequence_result_for_task,
)
pytestmark = pytest.mark.slow

@pytest.mark.parametrize("case", LOGIC_GATE_CASES, ids=lambda case: case.name)
def test_logic_gate_examples_are_monata_digital_test_cases(case: LogicGateCase):
    schematic = case.circuit().ensure_built()
    table = DigitalTruthTable(
        case.circuit,
        inputs=case.inputs,
        outputs=case.outputs,
        expected=case.expected,
        period=1.0,
        step=0.1,
        load_cap="1f",
        metadata={"fixture": "logic-gate"},
    )

    tasks = table.transient_tasks()
    results = [_sequence_result_for_task(table, tasks[0])]
    rows = table.extract_transient_results(results)

    assert len(schematic.pdk_instances) == case.pdk_instance_count
    assert len(tasks) == 1
    assert all(isinstance(task.analysis_spec, TranSpec) for task in tasks)
    assert tasks[-1].metadata["fixture"] == "logic-gate"
    assert "digital_truth_table" not in tasks[-1].metadata
    payload = _digital_task_metadata(tasks[-1])
    assert payload["digital_truth_table"]["dut"] == case.name
    assert payload["digital_truth_table"]["task_kind"] == "digital-single-bit-arc-sequence"
    assert payload["measurements"] == ["truth_table"]
    assert payload["stimulus"]["kind"] == "digital_single_bit_arc_sequence"
    assert payload["stimulus"]["vectors"] == len(case.expected.rows)
    assert payload["stimulus"]["total_arcs"] == len(case.expected.rows) * len(case.inputs)
    dut_line = f"Xdut {' '.join((*case.inputs, *case.outputs, 'vdd', '0'))} {case.name}"

    assert dut_line in render_ngspice(tasks[-1].circuit)
    assert rows.failed == []
    assert [
        (row["inputs"], row["actual"], row["expected"], row["status"])
        for row in rows.as_dicts()
    ] == [
        (inputs, expected, expected, "PASS")
        for inputs, expected in (
            (bits_to_text(bits), bits_to_text(case.expected(bits)))
            for bits in table.combinations()
        )
    ]


def test_digital_arc_chunk_size_validation_and_defaults():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        period=2e-9,
        transition=1e-10,
    )

    sequence_chunks = table.single_bit_arc_sequence_chunks(chunk_size=3)
    delay_chunks = table.propagation_delay_arc_chunks(chunk_size=2)

    assert [len(chunk) for chunk in sequence_chunks] == [3, 3, 2]
    assert [len(chunk) for chunk in delay_chunks] == [2, 2]
    with pytest.raises(ValueError, match="single-bit arc chunk_size must be positive"):
        table.single_bit_arc_sequence_chunks(chunk_size=0)
    with pytest.raises(ValueError, match="propagation delay chunk_size must be positive"):
        table.propagation_delay_arc_chunks(chunk_size=0)


def test_expected_table_data_structure_drives_exact_comparison():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        period=1.0,
        step=0.1,
    )
    results = [
        SimResult(status="ok", sweep_var=np.array([0.0, 1.0]), waveforms={"out": np.array([0.0, value])}, corner=None)
        for value in (0.0, 0.0, 0.0, 1.0)
    ]

    truth_table = table.extract_transient_results(results)

    assert [row.as_dict()["expected"] for row in truth_table] == ["0", "0", "0", "1"]
    assert truth_table.failed == []


def test_config_resolution_normalizes_inputs_and_preserves_metadata_contract():
    table = DigitalTruthTable(
        And2,
        inputs=["a", "b"],
        outputs=["out"],
        dependencies=[HelperCell],
        rails=("vdd", "gnd"),
        complement_inputs=["na", "nb"],
        vdd=1.2,
        expected=_expected_and,
        metadata={"oracle": "observed", "owner": "unit-test"},
        backend_options={"rawfile_format": "binary"},
        artifacts="artifacts",
    )

    assert table.inputs == ("a", "b")
    assert table.outputs == ("out",)
    assert table.dependencies[0] is HelperCell
    assert table.complement_inputs == ("na", "nb")
    assert table.threshold == pytest.approx(0.6)
    assert table.claim.as_dict() == OBSERVED_CLAIM
    assert table.metadata["oracle"] == "observed"
    assert table.metadata["owner"] == "unit-test"
    assert table.metadata["claim"] == OBSERVED_CLAIM
    assert table.backend_options == {"rawfile_format": "binary"}
    assert str(table.artifacts.directory) == "artifacts"
    assert table.tolerance is None
    assert table.comparator is None

    task = table.transient_tasks()[0]

    assert task.output_names == ("a", "b", "out")
    assert task.metadata["oracle"] == "observed"
    assert task.metadata["owner"] == "unit-test"
    assert _digital_task_metadata(task)["digital_truth_table"]["threshold"] == pytest.approx(0.6)
    assert task.backend_options == {"rawfile_format": "binary"}
    assert str(task.artifacts.directory) == "artifacts"


def test_digital_task_metadata_preserves_monata_namespace_mapping():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        metadata={"monata": {"owner": "custom-tool"}},
    )

    task = table.transient_tasks()[0]

    assert task.metadata["monata"]["owner"] == "custom-tool"
    assert _digital_task_metadata(task)["schema"] == "monata.sim.digital-task.v1"


def test_digital_task_metadata_rejects_non_mapping_monata_namespace():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        metadata={"monata": "custom-tool"},
    )

    with pytest.raises(ValueError, match=r"metadata\['monata'\]"):
        table.transient_tasks()


def test_transient_observation_rejects_unknown_payload_fields():
    with pytest.raises(ValueError, match="unknown digital transient observation fields: unexpected"):
        DigitalTransientObservation.from_dict({"stop": 1e-8, "unexpected": True})


def test_digital_truth_table_mode_rejects_removed_vector_transient_alias():
    assert resolve_digital_truth_table_mode("transient") == "transient"
    with pytest.raises(ValueError, match="unsupported digital truth-table mode: vector-transient"):
        resolve_digital_truth_table_mode("vector-transient")


def test_digital_truth_table_rejects_removed_op_load_resistance():
    with pytest.raises(TypeError, match="op_load_resistance"):
        DigitalTruthTable(
            And2,
            inputs=("a", "b"),
            outputs=("out",),
            op_load_resistance="1e12",  # type: ignore[reportCallIssue]
        )


def test_complement_inputs_must_match_inputs():
    with pytest.raises(ValueError, match="complement_inputs"):
        DigitalTruthTable(
            And2,
            inputs=("a", "b"),
            outputs=("out",),
            complement_inputs=("a_bar",),
        )
