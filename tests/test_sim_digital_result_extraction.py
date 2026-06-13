from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

from monata.sim.core import SimResult
from monata.sim.digital_table import (
    DigitalTruthTable,
)
from support.digital_cases import (
    And2,
    EXACT_CLAIM,
    INV_EXPECTED_TABLE,
    Inverter,
    _expected_and,
    _sequence_result_for_task,
)
pytestmark = pytest.mark.slow

def test_digital_result_exports_measurement_payloads():
    table = DigitalTruthTable(
        Inverter,
        inputs=("vin",),
        outputs=("out",),
        expected=INV_EXPECTED_TABLE,
        period=1.0,
        step=0.01,
        transition=0.1,
    )
    vector_results = [
        SimResult(
            status="ok",
            sweep_var=np.array([0.0, 1.0]),
            waveforms={"out": np.array([float(bit), float(int(not bit))])},
            corner=None,
        )
        for bit in (0, 1)
    ]
    timing_task = table.propagation_delay_task()
    timing_result = _sequence_result_for_task(table, timing_task, delay=(0.2, 0.3))
    truth_table = table.extract_transient_results(vector_results)
    result = truth_table.with_propagation_delays(
        table.extract_propagation_delays(timing_result),
        sim_result=timing_result,
    )

    measurements = result.measurements_as_dict()
    truth_table_measurement = cast(Mapping[str, object], measurements["truth_table"])
    delay_measurement = cast(Mapping[str, object], measurements["max_propagation_delay"])

    assert truth_table_measurement["status"] == "PASS"
    assert truth_table_measurement["rows"] == 2
    assert delay_measurement["value"] == pytest.approx(0.3)
    assert delay_measurement["unit"] == "s"


def test_extract_transient_results_samples_truth_table_rows():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=_expected_and,
        period=1.0,
        step=0.1,
    )
    results = [
        SimResult(status="ok", sweep_var=np.array([0.0, 1.0]), waveforms={"out": np.array([0.0, value])}, corner=None)
        for value in (0.0, 0.0, 0.0, 1.0)
    ]

    truth_table = table.extract_transient_results(results)

    assert [row.as_dict() for row in truth_table] == [
        {
            "inputs": "00",
            "actual": "0",
            "status": "PASS",
            "expected": "0",
            "claim": EXACT_CLAIM,
            "sample_time": 1.0,
            "samples": {"out": 0.0},
        },
        {
            "inputs": "01",
            "actual": "0",
            "status": "PASS",
            "expected": "0",
            "claim": EXACT_CLAIM,
            "sample_time": 1.0,
            "samples": {"out": 0.0},
        },
        {
            "inputs": "10",
            "actual": "0",
            "status": "PASS",
            "expected": "0",
            "claim": EXACT_CLAIM,
            "sample_time": 1.0,
            "samples": {"out": 0.0},
        },
        {
            "inputs": "11",
            "actual": "1",
            "status": "PASS",
            "expected": "1",
            "claim": EXACT_CLAIM,
            "sample_time": 1.0,
            "samples": {"out": 1.0},
        },
    ]
    assert truth_table.failed == []


def test_extract_transient_results_samples_digital_sequence():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=_expected_and,
        period=1.0,
        step=0.1,
    )
    task = table.transient_tasks()[0]
    result = _sequence_result_for_task(table, task)

    truth_table = table.extract_transient_results([result])

    assert [
        (row.as_dict()["inputs"], row.as_dict()["actual"], row.as_dict()["status"])
        for row in truth_table
    ] == [("00", "0", "PASS"), ("01", "0", "PASS"), ("10", "0", "PASS"), ("11", "1", "PASS")]
    assert all(row.sample_time is not None for row in truth_table)
    assert truth_table.sim_results == (result,)
    assert truth_table.failed == []


def test_extract_transient_results_uses_one_result_per_vector():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=_expected_and,
        threshold=0.5,
    )
    results = [
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None),
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([1.0])}, corner=None),
    ]

    truth_table = table.extract_transient_results(results)

    assert truth_table.mode == "transient"
    assert [row.as_dict()["status"] for row in truth_table] == ["PASS", "PASS", "PASS", "PASS"]


def test_vector_extractors_reject_partial_or_extra_result_counts():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=_expected_and,
        threshold=0.5,
    )
    one_result = [
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None)
    ]
    five_results = [
        SimResult(status="ok", sweep_var=None, waveforms={"out": np.array([0.0])}, corner=None)
        for _ in range(5)
    ]
    transient_results = [
        SimResult(status="ok", sweep_var=np.array([0.0]), waveforms={"out": np.array([0.0])}, corner=None)
    ]

    with pytest.raises(RuntimeError, match="transient result count mismatch: expected=4, got=1"):
        table.extract_transient_results(one_result)
    with pytest.raises(RuntimeError, match="transient result count mismatch: expected=4, got=5"):
        table.extract_transient_results(five_results)
    with pytest.raises(RuntimeError, match="transient result count mismatch: expected=4, got=1"):
        table.extract_transient_results(transient_results)


def test_extract_transient_results_samples_final_values():
    table = DigitalTruthTable(
        And2,
        inputs=("a", "b"),
        outputs=("out",),
        expected=_expected_and,
        threshold=0.5,
    )
    results = [
        SimResult(status="ok", sweep_var=np.array([0.0, 1.0]), waveforms={"out": np.array([0.2, 0.0])}, corner=None),
        SimResult(status="ok", sweep_var=np.array([0.0, 1.0]), waveforms={"out": np.array([0.2, 0.0])}, corner=None),
        SimResult(status="ok", sweep_var=np.array([0.0, 1.0]), waveforms={"out": np.array([0.2, 0.0])}, corner=None),
        SimResult(status="ok", sweep_var=np.array([0.0, 1.0]), waveforms={"out": np.array([0.2, 1.0])}, corner=None),
    ]

    truth_table = table.extract_transient_results(results)

    assert truth_table.mode == "transient"
    assert [row.as_dict()["status"] for row in truth_table] == ["PASS", "PASS", "PASS", "PASS"]
    assert truth_table.as_dicts()[-1]["sample_time"] == 1.0
