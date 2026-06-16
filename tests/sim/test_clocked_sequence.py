"""Tests for clock-driven Gray-code digital simulation mode."""

from __future__ import annotations

import pytest

from monata.netlist import render_ngspice
from monata.sim._digital_bits import (
    gray_code_bit_flip,
    gray_code_chunks,
    gray_code_sequence,
)
from monata.sim.digital_verify import DigitalWaveformAnalyzer

from support.digital_cases import (
    And2,
    AND2_EXPECTED_TABLE,
    _digital_task_metadata,
    _make_stimulus,
    _sequence_result_for_task,
)

pytestmark = pytest.mark.slow


# ── Gray code generator ──────────────────────────────────────────────


def test_gray_code_width_1():
    seq = gray_code_sequence(1)
    assert seq == ((0,), (1,))


def test_gray_code_width_2():
    seq = gray_code_sequence(2)
    assert seq == ((0, 0), (0, 1), (1, 1), (1, 0))


def test_gray_code_width_3_all_unique():
    seq = gray_code_sequence(3)
    assert len(seq) == 8
    assert len(set(seq)) == 8


def test_gray_code_single_bit_transitions():
    for w in range(1, 6):
        seq = gray_code_sequence(w)
        flips = gray_code_bit_flip(seq)
        assert len(flips) == len(seq) - 1


def test_gray_code_deterministic():
    assert gray_code_sequence(4) == gray_code_sequence(4)


def test_gray_code_chunks_single():
    chunks = gray_code_chunks(3)
    assert len(chunks) == 1
    assert chunks[0][0] == 0
    assert chunks[0][1] == (0, 0, 0)
    assert len(chunks[0][2]) == 7


def test_gray_code_chunks_with_slots():
    chunks = gray_code_chunks(3, slots_per_chunk=3)
    assert len(chunks) == 3
    for i in range(len(chunks) - 1):
        last_of_prev = chunks[i][2][-1] if chunks[i][2] else chunks[i][1]
        first_of_next = chunks[i + 1][1]
        assert last_of_prev == first_of_next, f"chunk {i}→{i+1} gap"
    total_transitions = sum(len(c[2]) for c in chunks)
    assert total_transitions == 7


def test_gray_code_bit_flip_empty():
    assert gray_code_bit_flip(((),)) == ()
    assert gray_code_bit_flip(((0,),)) == ()


# ── Clocked circuit builder ──────────────────────────────────────────


def test_clocked_circuit_has_clock_and_dut():
    stim = _make_stimulus(And2, inputs=("a", "b"), outputs=("out",),
                          period=4e-9, transition=1e-9)
    states = gray_code_sequence(2)
    circuit = stim.clocked_sequence_circuit(
        states, initial_settle=5e-8, clock_period=4e-9,
    )
    text = render_ngspice(circuit)
    assert "Vclk clk 0 PULSE(" in text
    assert "Xdut a b out vdd 0 and2" in text


def test_clocked_circuit_single_state():
    stim = _make_stimulus(And2, inputs=("a",), outputs=("out",),
                          period=4e-9, transition=1e-9)
    circuit = stim.clocked_sequence_circuit(((0,),), initial_settle=5e-8, clock_period=4e-9)
    assert "Vclk clk 0 PULSE(" in render_ngspice(circuit)


# ── Task construction ────────────────────────────────────────────────


def test_clocked_tasks_build_metadata():
    stim = _make_stimulus(And2, inputs=("a", "b"), outputs=("out",),
                          period=4e-9, step=5e-12, transition=1e-9)
    tasks = stim.build_tasks(
        initial_settle=5e-8,
        measurements=("truth_table", "max_propagation_delay"),
        clock_period=4e-9, slots_per_task=2,
    )
    assert len(tasks) == 2
    for task in tasks:
        payload = _digital_task_metadata(task)
        assert payload["digital_verification"]["task_kind"] == "digital-sequence"
        assert payload["stimulus"]["kind"] == "digital_sequence"
    assert "clk" in tasks[0].output_names


def test_clocked_tasks_single_chunk_for_small_width():
    stim = _make_stimulus(And2, inputs=("a", "b"), outputs=("out",),
                          period=4e-9, step=5e-12)
    tasks = stim.build_tasks(initial_settle=5e-8, measurements=("truth_table",))
    assert len(tasks) == 1


# ── Truth-table extraction ───────────────────────────────────────────


def test_clocked_extraction_matches_expected_and2():
    stim = _make_stimulus(And2, inputs=("a", "b"), outputs=("out",),
                          period=4e-9, step=1e-10, transition=2e-10)
    tasks = stim.build_tasks(initial_settle=1e-8, measurements=("truth_table",), clock_period=4e-9)
    assert len(tasks) == 1
    result = _sequence_result_for_task(stim, tasks[0], delay=0.0)
    analyzer = DigitalWaveformAnalyzer(_make_spec(And2))
    extracted = analyzer.verify([result], measurements=("truth_table",), vdd=1.0, sample_fraction=0.9)
    assert len(extracted) == 4
    assert extracted.failed == []
    assert all(row.passed for row in extracted)


def test_clocked_extraction_delay_from_clocked_sequence():
    stim = _make_stimulus(And2, inputs=("a", "b"), outputs=("out",),
                          period=4e-9, step=1e-10, transition=2e-10)
    tasks = stim.build_tasks(initial_settle=1e-8, measurements=("max_propagation_delay",), clock_period=4e-9)
    analyzer = DigitalWaveformAnalyzer(_make_spec(And2))
    extracted = analyzer.verify(
        [_sequence_result_for_task(stim, tasks[0], delay=2e-10)],
        measurements=("max_propagation_delay",), vdd=1.0, sample_fraction=0.9,
    )
    assert extracted.max_propagation_delay is not None
    assert extracted.max_propagation_delay == pytest.approx(2e-10)


def test_clocked_chunks_merge_consistently():
    stim = _make_stimulus(And2, inputs=("a", "b"), outputs=("out",),
                          period=4e-9, step=1e-10, transition=2e-10)
    tasks = stim.build_tasks(initial_settle=1e-8, measurements=("truth_table",),
                             clock_period=4e-9, slots_per_task=2)
    results = [_sequence_result_for_task(stim, task, delay=0.0) for task in tasks]
    analyzer = DigitalWaveformAnalyzer(_make_spec(And2))
    extracted = analyzer.verify(results, measurements=("truth_table",), vdd=1.0, sample_fraction=0.9)
    assert len(extracted) == 4
    assert extracted.failed == []


# ── Public API ───────────────────────────────────────────────────────


def test_transient_tasks_uses_digital_sequence_kind():
    stim = _make_stimulus(And2, inputs=("a", "b"), outputs=("out",),
                          period=4e-9, step=5e-12)
    tasks = stim.build_tasks(initial_settle=5e-8, measurements=("truth_table",))
    payload = _digital_task_metadata(tasks[0])
    assert payload["digital_verification"]["task_kind"] == "digital-sequence"


def test_run_transient_produces_valid_result():
    stim = _make_stimulus(And2, inputs=("a", "b"), outputs=("out",),
                          period=4e-9, step=1e-10, transition=2e-10)
    tasks = stim.build_tasks(initial_settle=1e-8, measurements=("truth_table",), clock_period=4e-9)
    result = _sequence_result_for_task(stim, tasks[0], delay=0.0)
    analyzer = DigitalWaveformAnalyzer(_make_spec(And2))
    extracted = analyzer.verify([result], measurements=("truth_table",), vdd=1.0, sample_fraction=0.9)
    assert len(extracted) == 4
    assert extracted.failed == []


# ── Helpers ──────────────────────────────────────────────────────────


def _make_spec(dut_cls):
    """Build a minimal verification spec for test purposes."""
    from monata.sim.digital_spec import DigitalVerificationSpec, DigitalVerificationMeasure
    return DigitalVerificationSpec(
        dut=dut_cls.NAME,
        inputs=("a", "b"),
        outputs=("out",),
        measures=(DigitalVerificationMeasure(name="truth_table", oracle="exact",
                                              expected=AND2_EXPECTED_TABLE),),
    )
