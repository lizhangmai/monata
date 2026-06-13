from typing import Any, cast

import numpy as np
import pytest

from monata.netlist import render_ngspice
from monata.sim.core import SimResult, TranSpec
from monata.sim.digital_claims import (
    DigitalComparisonContext,
    DigitalComparisonResult,
    DigitalOutputTolerance,
    DigitalVerificationClaim,
)
from monata.sim.digital_table import (
    DigitalTruthTable,
    bits_to_text,
)
from support.digital_cases import (
    AND2_EXPECTED_TABLE,
    And2,
    CUSTOM_CLAIM,
    INV_EXPECTED_TABLE,
    Inverter,
    OBSERVED_CLAIM,
    TOLERANCED_CLAIM,
    XOR2_EXPECTED_TABLE,
    _digital_task_metadata,
    _expected_and,
    _sequence_result_for_task,
)
pytestmark = pytest.mark.slow

def test_propagation_delay_arcs_follow_state_transitions_when_possible():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=XOR2_EXPECTED_TABLE,
        period=1.0,
        transition=0.1,
    )

    arcs = table.propagation_delay_arcs()

    assert len(arcs) == 8
    assert all(before.to_inputs == after.from_inputs for before, after in zip(arcs, arcs[1:]))


def test_propagation_delay_task_builds_single_input_transition_arcs():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        period=2e-9,
        step=5e-11,
        transition=1e-10,
        load_cap="1f",
    )

    task = table.propagation_delay_task()
    text = render_ngspice(task.circuit)

    assert isinstance(task.analysis_spec, TranSpec)
    assert task.analysis_spec.stop == pytest.approx(36e-9)
    assert task.output_names == ("a", "b", "out")
    assert "measurements" not in task.metadata
    payload = _digital_task_metadata(task)
    assert payload["digital_truth_table"]["task_kind"] == "digital-single-bit-arc-sequence"
    assert payload["measurements"] == ["max_propagation_delay"]
    assert payload["stimulus"]["kind"] == "digital_single_bit_arc_sequence"
    assert payload["stimulus"]["arc_coverage"] == "directed_single_bit_exhaustive"
    assert payload["stimulus"]["arcs"] == 8
    assert payload["stimulus"]["measurable_arcs"] == 4
    assert "Va a 0 PWL(" in text
    assert "Vb b 0 PWL(" in text
    assert "Xdut a b out vdd 0 and2" in text
    assert "Cload_out out 0 1f" in text


def test_digital_skew_step_rejects_negative_or_too_large_timing():
    with pytest.raises(ValueError, match="skew_step"):
        DigitalTruthTable(
            And2,
            inputs=("a", "b"),
            outputs=("out",),
            expected=AND2_EXPECTED_TABLE,
            period=2e-9,
            transition=1e-10,
            skew_step=-1e-10,
        )

    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        period=2e-9,
        cycles_per_vector=1,
        transition=1.8e-9,
        skew_step=0.3e-9,
    )

    with pytest.raises(RuntimeError, match="skewed input timing|transition timing"):
        table.propagation_delay_task()


def test_extract_propagation_delay_reports_max_arc():
    table = DigitalTruthTable(
        Inverter,
        inputs=("vin",),
        outputs=("out",),
        expected=INV_EXPECTED_TABLE,
        period=1.0,
        step=0.01,
        transition=0.1,
    )
    delays = (0.2, 0.3)
    task = table.propagation_delay_task()
    result = _sequence_result_for_task(table, task, delay=delays)

    rows = table.extract_propagation_delays(result)

    assert len(rows) == 2
    assert [row.delay for row in rows] == pytest.approx(list(delays))
    assert max(rows, key=lambda row: row.delay).as_dict() == {
        "from_inputs": bits_to_text(rows[1].from_inputs),
        "to_inputs": bits_to_text(rows[1].to_inputs),
        "input": "vin",
        "output": "out",
        "input_edge": rows[1].input_edge,
        "output_edge": rows[1].output_edge,
        "input_crossing": pytest.approx(rows[1].input_crossing),
        "output_crossing": pytest.approx(rows[1].output_crossing),
        "delay": pytest.approx(0.3),
    }


def test_config_resolution_infers_toleranced_oracle_from_tolerance():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=_expected_and,
        tolerance=0.05,
    )

    assert table.claim.as_dict() == TOLERANCED_CLAIM
    assert table.metadata["oracle"] == "toleranced"
    assert table.metadata["claim"] == TOLERANCED_CLAIM
    assert isinstance(table.comparator, DigitalOutputTolerance)


def test_observed_rows_have_canonical_non_exact_claim():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        oracle="observed",
        threshold=0.5,
    )
    results = [
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([1.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([1.0])}, corner=None),
    ]

    truth_table = table.extract_transient_results(results)
    rows = truth_table.as_dicts()

    assert truth_table.claim is not None
    assert truth_table.claim.as_dict() == OBSERVED_CLAIM
    assert rows[0]["status"] == "OBSERVED_NON_EXACT"
    assert rows[0]["claim"] == OBSERVED_CLAIM
    assert "expected" not in rows[0]
    assert truth_table.failed == []


def test_claim_from_dict_rejects_legacy_source_observation():
    payload = {
        "oracle": "source-observation",
        "claim_strength": "observation",
        "assertion": "source outputs were observed for each input vector",
        "expected_required": False,
        "correctness_claim": "none",
    }

    assert DigitalVerificationClaim.from_oracle("exact").oracle == "exact"
    with pytest.raises(ValueError, match="unsupported digital verification claim oracle"):
        DigitalVerificationClaim.from_dict(payload)


def test_claim_from_dict_rejects_unknown_payload_fields():
    payload = {
        "oracle": "observed",
        "claim_strength": "observation",
        "assertion": "source outputs were observed for each input vector",
        "expected_required": False,
        "correctness_claim": "none",
        "unexpected": True,
    }

    with pytest.raises(ValueError, match="unknown digital verification claim fields: unexpected"):
        DigitalVerificationClaim.from_dict(payload)


@pytest.mark.parametrize("oracle", ["observation", "tolerance", "custom-comparator", "source-observation"])
def test_claim_from_oracle_rejects_noncanonical_aliases(oracle: str):
    with pytest.raises(ValueError, match="unsupported digital verification oracle"):
        DigitalVerificationClaim.from_oracle(oracle)


def test_toleranced_oracle_uses_output_voltage_samples():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=_expected_and,
        oracle="toleranced",
        tolerance=0.15,
        threshold=0.5,
        vdd=1.0,
    )
    results = [
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.2])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.9])}, corner=None),
    ]

    truth_table = table.extract_transient_results(results)
    rows = truth_table.as_dicts()

    assert truth_table.claim is not None
    assert truth_table.claim.as_dict() == TOLERANCED_CLAIM
    assert [row["status"] for row in rows] == ["FAIL", "PASS", "PASS", "PASS"]
    first_comparison = cast(dict[str, Any], rows[0]["comparison"])
    first_details = cast(dict[str, Any], first_comparison["details"])
    first_out_details = cast(dict[str, Any], first_details["out"])
    last_comparison = cast(dict[str, Any], rows[-1]["comparison"])

    assert rows[0]["claim"] == TOLERANCED_CLAIM
    assert first_comparison["matched"] is False
    assert first_comparison["reason"] == "outside_tolerance"
    assert first_out_details["delta"] == 0.2
    assert last_comparison["reason"] == "within_tolerance"
    assert len(truth_table.failed) == 1


def test_toleranced_oracle_fails_shape_mismatch_instead_of_truncating():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out0", "out1"),
        expected=lambda _bits: (1,),
        oracle="toleranced",
        tolerance=0.2,
        threshold=0.5,
        vdd=1.0,
    )
    results = [
        SimResult(
            status="ok",
            sweep_var=None,
            waveforms={"out0": np.array([1.0]), "out1": np.array([0.0])},
            corner=None,
        )
        for _ in range(4)
    ]

    truth_table = table.extract_transient_results(results)
    row = truth_table.as_dicts()[0]
    comparison = cast(dict[str, Any], row["comparison"])

    assert row["status"] == "FAIL"
    assert comparison == {
        "matched": False,
        "reason": "shape_mismatch",
        "details": {"actual_count": 2, "expected_count": 1, "output_count": 2},
    }


def test_custom_comparator_is_first_class_row_oracle():
    seen_contexts = []

    def only_last_vector(context: DigitalComparisonContext) -> DigitalComparisonResult:
        seen_contexts.append(context)
        return DigitalComparisonResult(context.inputs == (1, 1), reason="only_last_vector")

    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=_expected_and,
        oracle="custom",
        comparator=only_last_vector,
        threshold=0.5,
    )
    results = [
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
    ]

    truth_table = table.extract_transient_results(results)
    rows = truth_table.as_dicts()

    assert truth_table.claim is not None
    assert truth_table.claim.as_dict() == CUSTOM_CLAIM
    assert [row["status"] for row in rows] == ["FAIL", "FAIL", "FAIL", "PASS"]
    assert rows[-1]["actual"] == "0"
    assert rows[-1]["expected"] == "1"
    assert rows[-1]["comparison"] == {"matched": True, "reason": "only_last_vector"}
    assert len(seen_contexts) == 4
    assert seen_contexts[-1].claim.as_dict() == CUSTOM_CLAIM
    assert seen_contexts[-1].samples == {"out": 0.0}


def test_comparator_oracles_validate_required_configuration():
    with pytest.raises(ValueError, match="toleranced digital oracle requires tolerance or comparator"):
        DigitalTruthTable(And2, inputs=("a", "b"), outputs=("out",), expected=_expected_and, oracle="toleranced")

    with pytest.raises(ValueError, match="custom digital oracle requires comparator"):
        DigitalTruthTable(And2, inputs=("a", "b"), outputs=("out",), oracle="custom")

    inferred_custom = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        comparator=lambda _context: True,
    )
    assert inferred_custom.claim.as_dict() == CUSTOM_CLAIM

    with pytest.raises(ValueError, match="comparators require custom or toleranced oracle"):
        DigitalTruthTable(
            And2,
            inputs=("a", "b"),
            outputs=("out",),
            expected=_expected_and,
            oracle="exact",
            comparator=lambda _context: True,
        )

    with pytest.raises(ValueError, match="voltage_tolerance must be non-negative"):
        DigitalOutputTolerance(-0.1)


def test_propagation_delay_task_rejects_removed_source_shape():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=AND2_EXPECTED_TABLE,
        transition=0.1,
    )

    with pytest.raises(TypeError, match="source_shape"):
        table.propagation_delay_task(source_shape="exp")  # type: ignore[reportCallIssue]
